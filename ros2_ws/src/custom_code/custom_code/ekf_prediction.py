#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from custom_msgs.msg import Belief, DeltaOdom
from geometry_msgs.msg import Pose2D
import numpy as np


class EKFPrediction(Node):
    """Nodo que implementa el paso de PREDICCIÓN del EKF.

    - Se subscribe a /belief (Belief) para recibir el belief inicial y futuras correcciones.
    - Se subscribe a /delta (DeltaOdom) para recibir actualizaciones de odometría en formato deltas.
    - Publica en /belief (Belief) las predicciones del belief.
    """

    def __init__(self):
        super().__init__('ekf_prediction')

        # Parámetros
        self.declare_parameter('wheel_base', 0.16)  # Según me fijé, la distancia entre ruedas del turtlebot es 16 cm.
        self.declare_parameter('alpha1', 0.02)
        self.declare_parameter('alpha2', 0.02)
        self.declare_parameter('alpha3', 0.02)
        self.declare_parameter('alpha4', 0.02)

        self.wheel_base = self.get_parameter('wheel_base').get_parameter_value().double_value
        self.alpha1 = float(self.get_parameter('alpha1').get_parameter_value().double_value)
        self.alpha2 = float(self.get_parameter('alpha2').get_parameter_value().double_value)
        self.alpha3 = float(self.get_parameter('alpha3').get_parameter_value().double_value)
        self.alpha4 = float(self.get_parameter('alpha4').get_parameter_value().double_value)

        # Publicador y subscriptores
        self.belief_pub = self.create_publisher(Belief, '/belief', 10)
        self.belief_sub = self.create_subscription(Belief, '/belief', self._belief_callback, 10)
        self.delta_sub = self.create_subscription(DeltaOdom, '/delta', self._delta_callback, 10)

        # Estado interno: mu (x,y,theta) y Sigma (3x3)
        self.have_belief = False
        self.mu = np.zeros(3)
        self.Sigma = np.eye(3) * 1e-6  # valor por defecto pequeño

        self.get_logger().info('EKFPrediction inicializado (wheel_base=%.3f m)' % self.wheel_base)

    def _belief_callback(self, msg: Belief) -> None:
        """Recibe beliefs (inicial y correcciones). Actualiza el estado interno sin predecir."""
        self.mu[0] = msg.mu.x
        self.mu[1] = msg.mu.y
        self.mu[2] = msg.mu.theta

        try:
            cov_list = list(msg.covariance)
            if len(cov_list) != 9:
                raise ValueError('Belief.covariance length != 9')
            self.Sigma = np.array(cov_list).reshape((3, 3))
        except Exception as e:
            self.get_logger().warn('Error al leer covariance del belief: %s. Usando identidad pequeña.' % str(e))
            self.Sigma = np.eye(3) * 1e-6

        self.have_belief = True
        self.get_logger().info('Belief recibido: x=%.3f y=%.3f theta=%.3f' % (self.mu[0], self.mu[1], self.mu[2]))

    def _delta_callback(self, msg: DeltaOdom) -> None:
        """Recibe deltas de odometría y realiza el paso de predicción (si ya se tiene belief inicial).

        Implementa las ecuaciones:
            δ_trans, δ_rot1, δ_rot2
            μ̄ = μ + [δ_trans cos(θ+δ_rot1); δ_trans sin(...); δ_rot1+δ_rot2]
            G_t = una matriz 3x3
            V_t = otra matriz 3x3
            M_t = matriz de los alphas
            Σ̄ = G Σ G^T + V M V^T
        """
        if not self.have_belief:
            self.get_logger().warn('Delta recibido pero no hay belief inicial. Ignorando delta.')
            return

        delta_rot1 = float(msg.dr1)
        delta_rot2 = float(msg.dr2)
        delta_trans = float(msg.dt)

        theta = self.mu[2]
        theta_mid = theta + delta_rot1

        # Predicción del estado (ecuación 6 del pseudocódigo de Nacho)
        mu_bar = self.mu.copy()
        mu_bar[0] += delta_trans * np.cos(theta_mid)
        mu_bar[1] += delta_trans * np.sin(theta_mid)
        mu_bar[2] = self._normalize_angle(self.mu[2] + (delta_rot1 + delta_rot2))

        # Jacobiano G_t (3x3)
        G = np.eye(3)
        G[0, 2] = -delta_trans * np.sin(theta_mid)
        G[1, 2] = delta_trans * np.cos(theta_mid)

        # Jacobiano V_t (3x3) derivadas de g respecto a los ruídos [δ_rot1, δ_trans, δ_rot2]
        V = np.zeros((3, 3))
        # Row 0: d/d(delta_rot1), d/d(delta_trans), d/d(delta_rot2)
        V[0, 0] = -delta_trans * np.sin(theta_mid)
        V[0, 1] = np.cos(theta_mid)
        V[0, 2] = 0.0
        # Row 1:
        V[1, 0] = delta_trans * np.cos(theta_mid)
        V[1, 1] = np.sin(theta_mid)
        V[1, 2] = 0.0
        # Row 2:
        V[2, 0] = 1.0
        V[2, 1] = 0.0
        V[2, 2] = 1.0

        # M_t: covarianza del ruido de movimiento (3x3)
        M = np.zeros((3, 3))

        M[0, 0] = self.alpha1 * (delta_rot1 ** 2) + self.alpha2 * (delta_trans ** 2)
        M[1, 1] = self.alpha3 * (delta_trans ** 2) + self.alpha4 * (delta_rot1 ** 2 + delta_rot2 ** 2)
        M[2, 2] = self.alpha1 * (delta_rot2 ** 2) + self.alpha2 * (delta_trans ** 2)

        # Predicción de la covarianza: Sigma_gorrito = G Sigma G^T + V M V^T
        Sigma_bar = G.dot(self.Sigma).dot(G.T) + V.dot(M).dot(V.T)

        # Actualizamos estado interno
        self.mu = mu_bar
        self.Sigma = Sigma_bar

        # Publicamos belief predicho
        belief_msg = Belief()
        belief_msg.mu = Pose2D()
        belief_msg.mu.x = float(self.mu[0])
        belief_msg.mu.y = float(self.mu[1])
        belief_msg.mu.theta = float(self.mu[2])
        belief_msg.covariance = [float(x) for x in self.Sigma.reshape(9).tolist()]

        self.belief_pub.publish(belief_msg)

        self.get_logger().info('Predicción publicada: x=%.3f y=%.3f theta=%.3f' % (self.mu[0], self.mu[1], self.mu[2]))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normaliza el ángulo a [-pi, pi]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi


def main(args=None):
    rclpy.init(args=args)
    node = EKFPrediction()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

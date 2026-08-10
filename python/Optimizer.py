import numpy as np
from typing import List, Dict, Tuple

class AdamW:
    def __init__(
            self,
            params: List[np.ndarray],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.state = {
            'm': [np.zeros_like(p) for p in params],
            'v': [np.zeros_like(p) for p in params],
            'step': 0
        }

    def step(self, grads: List[np.ndarray]) -> None:
        self.state['step'] += 1
        step = self.state['step']

        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            param *= (1 - self.lr * self.weight_decay)

            self.state['m'][i] = self.beta1 * self.state['m'][i] + (1 - self.beta1) * grad
            self.state['v'][i] = self.beta2 * self.state['v'][i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.state['m'][i] / bias_correction1
            v_hat = self.state['v'][i] / bias_correction2

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


    def get_params(self) -> List[np.ndarray]:
        return self.params

    def get_state(self) -> Dict:
        return {
            'step': self.state['step'],
            'm': [m.copy() for m in self.state['m']],
            'v': [v.copy() for v in self.state['v']]
        }
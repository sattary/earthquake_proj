import torch


class Physics:
    """
    Static class containing Partial Differential Equation (PDE) residuals for PINN.
    """

    @staticmethod
    def momentum_balance_2d(model, x):
        """
        Compute physics residual for 2D Plane Stress Momentum Balance.

        Governing Equations:
        d(sxx)/dx + d(sxy)/dy = 0
        d(sxy)/dx + d(syy)/dy = 0

        Args:
            model (nn.Module): The neural network approximating the stress field.
            x (torch.Tensor): Input coordinates (N, 2).

        Returns:
            res_x (torch.Tensor): Residual for x-momentum equation.
            res_y (torch.Tensor): Residual for y-momentum equation.
        """
        x.requires_grad = True
        out = model(x)

        # Unpack outputs: [vx, vy, sxx, syy, sxy]
        # sxx is index 2, syy is index 3, sxy is index 4
        sxx = out[:, 2]
        syy = out[:, 3]
        sxy = out[:, 4]

        # Gradients
        grads_sxx = torch.autograd.grad(
            sxx, x, grad_outputs=torch.ones_like(sxx), create_graph=True
        )[0]
        grads_syy = torch.autograd.grad(
            syy, x, grad_outputs=torch.ones_like(syy), create_graph=True
        )[0]
        grads_sxy = torch.autograd.grad(
            sxy, x, grad_outputs=torch.ones_like(sxy), create_graph=True
        )[0]

        dsxx_dx = grads_sxx[:, 0]
        # dsxx_dy = grads_sxx[:, 1]

        # dsyy_dx = grads_syy[:, 0]
        dsyy_dy = grads_syy[:, 1]

        dsxy_dx = grads_sxy[:, 0]
        dsxy_dy = grads_sxy[:, 1]

        # Equilibrium Equations (ignoring body forces/gravity for now as per tectonic scale approx)
        res_x = dsxx_dx + dsxy_dy
        res_y = dsxy_dx + dsyy_dy

        return res_x, res_y

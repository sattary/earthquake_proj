import torch


class Physics:
    """
    Static class containing Partial Differential Equation (PDE) residuals for PINN.
    """

    @staticmethod
    def momentum_balance_2d(model, x):
        """
        Compute physics residual for 2D Plane Stress Momentum Balance (Non-Dimensionalized).
        Solves on the normalized domain [-1, 1].

        Args:
            model (nn.Module): Network.
            x (torch.Tensor): Input coordinates (N, 2) in normalized domain.
        """
        x.requires_grad = True
        out = model(x)

        # Unpack outputs: [vx, vy, sxx, syy, sxy]
        sxx = out[:, 2]
        syy = out[:, 3]
        sxy = out[:, 4]

        # Gradients wrt NORMALIZED coordinates
        grads_sxx = torch.autograd.grad(
            sxx, x, grad_outputs=torch.ones_like(sxx), create_graph=True
        )[0]
        grads_syy = torch.autograd.grad(
            syy, x, grad_outputs=torch.ones_like(syy), create_graph=True
        )[0]
        grads_sxy = torch.autograd.grad(
            sxy, x, grad_outputs=torch.ones_like(sxy), create_graph=True
        )[0]

        # Non-Dimensionalized Derivatives (No scaling)
        dsxx_dx = grads_sxx[:, 0]
        dsyy_dy = grads_syy[:, 1]
        dsxy_dx = grads_sxy[:, 0]
        dsxy_dy = grads_sxy[:, 1]

        # Equilibrium Equations
        res_x = dsxx_dx + dsxy_dy
        res_y = dsxy_dx + dsyy_dy

        return res_x, res_y

    @staticmethod
    def constitutive_2d(model, x, eta=1.0):
        """
        Compute Constitutive Law residual (Non-Dimensionalized).
        Assumes dimensionless variables where eta_eff ~ 1.
        """
        x.requires_grad = True
        out = model(x)

        # Unpack: [vx, vy, sxx, syy, sxy]
        vx = out[:, 0]
        vy = out[:, 1]
        sxx = out[:, 2]
        syy = out[:, 3]
        sxy = out[:, 4]

        # Velocity Gradients (Strain Rates) wrt Normalized Coords
        grads_vx = torch.autograd.grad(
            vx, x, grad_outputs=torch.ones_like(vx), create_graph=True
        )[0]
        grads_vy = torch.autograd.grad(
            vy, x, grad_outputs=torch.ones_like(vy), create_graph=True
        )[0]

        dvx_dx = grads_vx[:, 0]
        dvx_dy = grads_vx[:, 1]
        dvy_dx = grads_vy[:, 0]
        dvy_dy = grads_vy[:, 1]

        exx = dvx_dx
        eyy = dvy_dy
        exy = 0.5 * (dvx_dy + dvy_dx)

        res_xx = sxx - 2 * eta * exx
        res_yy = syy - 2 * eta * eyy
        res_xy = sxy - 2 * eta * exy

        return res_xx, res_yy, res_xy

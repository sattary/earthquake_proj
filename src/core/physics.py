import torch
from src.core.constants import S0, V0, G_ACCEL, RHO_CRUST


class Physics:
    """
    Static class containing Partial Differential Equation (PDE) residuals for PINN.
    All methods assume non-dimensionalized inputs and handle physical scaling internally.
    """

    @staticmethod
    def momentum_balance_2d(model, x):
        """
        Compute physics residual for 2D Plane Stress Momentum Balance (Non-Dimensionalized).
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

        dsxx_dx = grads_sxx[:, 0]
        dsyy_dy = grads_syy[:, 1]
        dsxy_dx = grads_sxy[:, 0]
        dsxy_dy = grads_sxy[:, 1]

        res_x = dsxx_dx + dsxy_dy
        res_y = dsxy_dx + dsyy_dy

        return res_x, res_y

    @staticmethod
    def constitutive_2d(model, x, eta=1.0):
        """
        Compute Constitutive Law residual (Non-Dimensionalized).
        """
        x.requires_grad = True
        out = model(x)

        vx = out[:, 0]
        vy = out[:, 1]
        sxx = out[:, 2]
        syy = out[:, 3]
        sxy = out[:, 4]

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

    @staticmethod
    def momentum_balance_3d(
        model, x, rho=RHO_CRUST, g=G_ACCEL, scale_x=1.0, scale_z=1.0, stress_scale=S0
    ):
        """
        3D Momentum Balance with Gravity (Scaled & Anisotropic).
        """
        x.requires_grad = True
        out = model(x)

        # Output Map: szz:5, syz:7, sxz:8
        sxx = out[:, 3] * stress_scale
        syy = out[:, 4] * stress_scale
        szz = out[:, 5] * stress_scale
        sxy = out[:, 6] * stress_scale
        syz = out[:, 7] * stress_scale
        sxz = out[:, 8] * stress_scale

        def get_grads(tensor, inputs):
            return torch.autograd.grad(
                tensor, inputs, grad_outputs=torch.ones_like(tensor), create_graph=True
            )[0]

        grads_sxx = get_grads(sxx, x)
        grads_syy = get_grads(syy, x)
        grads_szz = get_grads(szz, x)
        grads_sxy = get_grads(sxy, x)
        grads_syz = get_grads(syz, x)
        grads_sxz = get_grads(sxz, x)

        inv_sx = 1.0 / scale_x
        inv_sz = 1.0 / scale_z

        div_x = (
            grads_sxx[:, 0] * inv_sx
            + grads_sxy[:, 1] * inv_sx
            + grads_sxz[:, 2] * inv_sz
        )
        div_y = (
            grads_sxy[:, 0] * inv_sx
            + grads_syy[:, 1] * inv_sx
            + grads_syz[:, 2] * inv_sz
        )
        div_z = (
            grads_sxz[:, 0] * inv_sx
            + grads_syz[:, 1] * inv_sx
            + grads_szz[:, 2] * inv_sz
        )

        res_x = div_x
        res_y = div_y
        res_z = div_z + rho * g

        norm = 1.0 / (rho * g)
        return res_x * norm, res_y * norm, res_z * norm

    @staticmethod
    def constitutive_3d(
        model, x, eta=1e21, scale_x=1.0, scale_z=1.0, stress_scale=S0, vel_scale=V0
    ):
        """
        3D Viscous Constitutive Law (Scaled).
        """
        x.requires_grad = True
        out = model(x)

        vx = out[:, 0] * vel_scale
        vy = out[:, 1] * vel_scale
        vz = out[:, 2] * vel_scale

        sxx = out[:, 3] * stress_scale
        syy = out[:, 4] * stress_scale
        szz = out[:, 5] * stress_scale
        sxy = out[:, 6] * stress_scale
        syz = out[:, 7] * stress_scale
        sxz = out[:, 8] * stress_scale

        def get_grads(tensor, inputs):
            return torch.autograd.grad(
                tensor, inputs, grad_outputs=torch.ones_like(tensor), create_graph=True
            )[0]

        grads_vx = get_grads(vx, x)
        grads_vy = get_grads(vy, x)
        grads_vz = get_grads(vz, x)

        inv_sx = 1.0 / scale_x
        inv_sz = 1.0 / scale_z

        dvx_dx = grads_vx[:, 0] * inv_sx
        dvx_dy = grads_vx[:, 1] * inv_sx
        dvx_dz = grads_vx[:, 2] * inv_sz

        dvy_dx = grads_vy[:, 0] * inv_sx
        dvy_dy = grads_vy[:, 1] * inv_sx
        dvy_dz = grads_vy[:, 2] * inv_sz

        dvz_dx = grads_vz[:, 0] * inv_sx
        dvz_dy = grads_vz[:, 1] * inv_sx
        dvz_dz = grads_vz[:, 2] * inv_sz

        exx = dvx_dx
        eyy = dvy_dy
        ezz = dvz_dz

        exy = 0.5 * (dvx_dy + dvy_dx)
        eyz = 0.5 * (dvy_dz + dvz_dy)
        exz = 0.5 * (dvx_dz + dvz_dx)

        trace_e = exx + eyy + ezz
        e_xx = exx - trace_e / 3.0
        e_yy = eyy - trace_e / 3.0
        e_zz = ezz - trace_e / 3.0

        trace_s = sxx + syy + szz
        tau_xx = sxx - trace_s / 3.0
        tau_yy = syy - trace_s / 3.0
        tau_zz = szz - trace_s / 3.0
        tau_xy = sxy
        tau_yz = syz
        tau_xz = sxz

        res_xx = tau_xx - 2 * eta * e_xx
        res_yy = tau_yy - 2 * eta * e_yy
        res_zz = tau_zz - 2 * eta * e_zz
        res_xy = tau_xy - 2 * eta * exy
        res_yz = tau_yz - 2 * eta * eyz
        res_xz = tau_xz - 2 * eta * exz

        res_vol = trace_e * eta

        inv_S0 = 1.0 / stress_scale
        return (
            res_xx * inv_S0,
            res_yy * inv_S0,
            res_zz * inv_S0,
            res_xy * inv_S0,
            res_yz * inv_S0,
            res_xz * inv_S0,
            res_vol * inv_S0,
        )

    @staticmethod
    def traction_free_surface(model, x_surf, stress_scale=S0):
        """
        Computes traction-free residuals at the surface (T = sigma . n = 0).
        """
        out = model(x_surf)

        szz = out[:, 5] * stress_scale
        syz = out[:, 7] * stress_scale
        sxz = out[:, 8] * stress_scale

        return sxz / stress_scale, syz / stress_scale, szz / stress_scale

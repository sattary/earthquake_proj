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

    @staticmethod
    def momentum_balance_3d(
        model, x, rho=2700.0, g=9.81, scale_x=1.0, scale_z=1.0, S0=1e7
    ):
        """
        3D Momentum Balance with Gravity (Scaled & Anisotropic).
        Args:
            scale_x: Horizontal Length Scale (m).
            scale_z: Vertical Length Scale (m).
            S0: Characteristic Stress Scale (Pa).
        """
        x.requires_grad = True
        out = model(x)

        # Output Map: 0:vx, 1:vy, 2:vz, 3:sxx, 4:syy, 5:szz, 6:sxy, 7:syz, 8:sxz
        sxx = out[:, 3] * S0
        syy = out[:, 4] * S0
        szz = out[:, 5] * S0
        sxy = out[:, 6] * S0
        syz = out[:, 7] * S0
        sxz = out[:, 8] * S0

        # Compute Gradients wrt Normalized Coordinates
        # We need gradients of ALL stress components
        # Helper to get grads (N, 3)
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

        # Physical Derivatives: anisotropic scaling
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

        # Momentum Balance
        # X: div_sigma_x = 0
        # Y: div_sigma_y = 0
        # Z: div_sigma_z - rho * g = 0 (z is positive UP? No, typically Z is just coordinate)
        # If z is Depth (positive down), then gravity acts in +z direction: + rho*g
        # If z is Elevation (positive up), then gravity acts in -z direction: - rho*g
        # Our VelocityModel: dep is km (positive down).
        # We normalized z. If z_norm follows z_dep, then +z is down.
        # So Normalized Z +1 is Deep, -1 is Surface.
        # Gravity pulls Down (+z). So Force is +rho*g.
        # Equation: Divergence + BodyForce = 0 => DivSigma + rho*g = 0
        # Wait, usually DivSigma + f = rho*a. Static: DivSigma + f = 0.
        # Force is Gravity vector g_vec = (0, 0, g).
        # So equation is dSxz/dx + dSyz/dy + dSzz/dz + rho*g = 0.

        res_x = div_x
        res_y = div_y
        res_z = div_z + rho * g

        # Normalize (Non-dimensionalize) the Residuals
        # Scale by weight of the crust (rho*g) so that errors are fractional
        # and on the same order as tectonic stress gradients.
        norm = 1.0 / (rho * g)
        return res_x * norm, res_y * norm, res_z * norm

    @staticmethod
    def constitutive_3d(model, x, eta=1e21, scale_x=1.0, scale_z=1.0, S0=1e7, V0=1e-9):
        """
        3D Viscous Constitutive Law (Scaled).
        R_const = (1/S0) * (sigma - 2*eta*epsilon_dot)
        Args:
            S0: Characteristic Stress Scale (Pa).
            V0: Characteristic Velocity Scale (m/s).
        """
        x.requires_grad = True
        out = model(x)

        # 0:vx, 1:vy, 2:vz
        # Assume Network output 'v' is dimensionless (v_hat).
        # Physical Velocity v = v_hat * V0
        vx = out[:, 0] * V0
        vy = out[:, 1] * V0
        vz = out[:, 2] * V0

        # Stress
        sxx = out[:, 3] * S0
        syy = out[:, 4] * S0
        szz = out[:, 5] * S0
        sxy = out[:, 6] * S0
        syz = out[:, 7] * S0
        sxz = out[:, 8] * S0

        def get_grads(tensor, inputs):
            return torch.autograd.grad(
                tensor, inputs, grad_outputs=torch.ones_like(tensor), create_graph=True
            )[0]

        grads_vx = get_grads(vx, x)
        grads_vy = get_grads(vy, x)
        grads_vz = get_grads(vz, x)

        inv_sx = 1.0 / scale_x
        inv_sz = 1.0 / scale_z

        # Strain Rates (epsilon_dot)
        # e_ij = 0.5 * (dvi/dxj + dvj/dxi)
        # Note: derivatives wrt normalized coords need appropriate scaler

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

        # Deviatoric Strain Rates: e_ij = epsilon_dot_ij - (1/3)*tr(epsilon_dot)*delta_ij
        trace_e = exx + eyy + ezz
        e_xx = exx - trace_e / 3.0
        e_yy = eyy - trace_e / 3.0
        e_zz = ezz - trace_e / 3.0
        # Shear components are already deviatoric (trace doesn't affect off-diagonals)

        # Deviatoric Stress: tau_ij = sigma_ij - (1/3)*tr(sigma)*delta_ij
        trace_s = sxx + syy + szz
        tau_xx = sxx - trace_s / 3.0
        tau_yy = syy - trace_s / 3.0
        tau_zz = szz - trace_s / 3.0
        tau_xy = sxy
        tau_yz = syz
        tau_xz = sxz

        # Residuals: Tau - 2*eta*E_deviatoric
        res_xx = tau_xx - 2 * eta * e_xx
        res_yy = tau_yy - 2 * eta * e_yy
        res_zz = tau_zz - 2 * eta * e_zz
        res_xy = tau_xy - 2 * eta * exy
        res_yz = tau_yz - 2 * eta * eyz
        res_xz = tau_xz - 2 * eta * exz

        # Volumetric Incompressibility Constraint: div(v) = 0
        # Scale by eta to match stress units (Pa)
        res_vol = trace_e * eta

        # Normalize: Divide by S0
        inv_S0 = 1.0 / S0
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
    def traction_free_surface(model, x_surf, S0=1e7):
        """
        Computes traction-free residuals at the surface (T = sigma . n = 0).
        For a flat top surface at z=const, normal n = (0, 0, -1) [normalized]
        T_x = sigma_xz = 0
        T_y = sigma_yz = 0
        T_z = sigma_zz = 0
        Args:
            x_surf: (N, 3) points sampled at the surface (z=-1).
        """
        out = model(x_surf)

        # Physical Stresses
        # Output Map: szz:5, syz:7, sxz:8
        szz = out[:, 5] * S0
        syz = out[:, 7] * S0
        sxz = out[:, 8] * S0

        # Normalize by S0 to keep loss O(1)
        return sxz / S0, syz / S0, szz / S0

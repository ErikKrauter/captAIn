from ..utils.meta import Registry, build_from_cfg


MPC = Registry("mpc")  # Model predictive control
MFRL = Registry("mfrl")  # Model free RL
BRL = Registry("brl")  # Offline RL / Batch RL
MBRL = Registry("mbrl")  # model based RL (this includes VAT-Mart)


def build_agent(cfg, default_args=None):
    if cfg is None:
        return None
    for agent_type in [MPC, MFRL, BRL, MBRL]:
        if cfg["type"] in agent_type:
            return build_from_cfg(cfg, agent_type, default_args)
    return None

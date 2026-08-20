"""Policy subclasses (pi0/groot) with auxiliary losses, registered as custom --policy.type.

Importing this package registers the configs, which is what has to happen *before*
the CLI is parsed: draccus resolves ``--policy.type=h2r_pi0`` against
``PreTrainedConfig``'s registry, and an unregistered name is a parse error.

Only the configuration modules are imported here. The modeling modules pull in
transformers and the model code, and LeRobot's plugin path imports them by name
when a policy is actually constructed -- so importing them here would load pi0 and
GR00T into every process that merely parses arguments.
"""

from h2r_il.policies.configuration_h2r_groot import H2RGrootConfig
from h2r_il.policies.configuration_h2r_pi0 import H2RPi0Config

__all__ = ["H2RGrootConfig", "H2RPi0Config"]

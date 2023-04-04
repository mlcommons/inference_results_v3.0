# Instructions about the power config files

## Power server config files

The `server*.cfg` files are the config files for the Power Server machine. They are needed to run the `server.py` script on the Power Server.
They specify the NTP server IP, PTD daemon path, power meter type, channel index, and power meter IP.

## NVPower model config files

The `nvpmodel*.conf` files are the config files to set the MaxQ mode on AGX Xavier or Xavier NX boards.
They are based on the `/etc/nvpmodel.conf` file shipped together with JetPack installation, plus an additional power mode config for MaxQ mode at the end of the file.

The fields in the `nvpmodel.conf` files are as follows:
- `CPU_ONLINE CORE_N 0|1`: Whether a CPU core should be enabled or disabled. At least 2 CPU cores should be enabled.
- `TPC_POWER_GATING TPC_PG_MASK 1`: Don't touch.
- `GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on`: Don't touch.
- `CPU_DENVER_0 MIN_FREQ|MAX_FREQ N`: The minimal and maximal clock frequencies for the CPUs.
- `GPU MIN_FREQ|MAX_FREQ N`: The minimal and maximal clock frequencies for the GPUs.
- `EMC MAX_FREQ N`: The maximal frequency for the memory clock.
- `DLA_CORE MAX_FREQ N`: The maximal frequency for the DLAs.
- `DLA_FALCON MAX_FREQ 640000000`: Don't touch.
- `PVA_VPS MAX_FREQ 819200000`: Don't touch. PVA is not used in MLPerf-Inference submission and will be automatically power-gated.
- `PVA_CORE MAX_FREQ 601600000`: Don't touch. PVA is not used in MLPerf-Inference submission and will be automatically power-gated.
- `CVNAS MAX_FREQ 576000000`: Don't touch.

# Boot/BIOS Firmware Settings

## AMD CBS

### NBIO Common Options
#### SMU Common Options
##### Determinism Control: Manual
##### Determinism Slider: Power
##### cTDP: Auto
##### DF Cstates: Auto

### DF Common Options

#### Scrubber
##### DRAM scrub time: Disabled
##### Poisson scrubber control: Disabled
##### Redirect scrubber control: Disabled

#### Memory Addressing
##### NUMA nodes per socket: NPS1
##### ACPI SRAT L3 Cche As NUMA Domain: Disabled

### CPU Common Options
#### Performance
##### SMT Control: Disable
#### Global C-state Control: Enabled

# Management Firmware Settings

Out-of-the-box.

# Power Management Settings

## Fan Speed

The fan speed is controlled through a variable called `pre_fan`.
For this system, workload and scenario this variable was set to `75` (6,750 RPM) as follows:

<pre>
sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>75</b> 0xFF
</pre>

## Maximum Frequency

The maximum chip frequency is controlled through a variable called `vc`.
For this system, workload and scenario this variable was set to `11`.


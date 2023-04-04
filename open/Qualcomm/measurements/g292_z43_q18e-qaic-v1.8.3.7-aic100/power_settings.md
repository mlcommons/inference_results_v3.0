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
##### SMT Control: Enable
#### Global C-state Control: Enabled

# Management Firmware Settings

Out-of-the-box.

# Fan Settings (8,100 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>100</b> 0xFF
 0a 3c 00
</pre>

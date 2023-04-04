# Boot/BIOS Firmware Settings

## AMD CBS

### NBIO Common Options
#### SMU Common Options
##### Determinism Slider = Power
##### cTDP = 280 W
##### Package Power Limit = 280 W
##### DF Cstates = Disabled
##### Fixed SOC Pstate = P0

### DF Common Options
#### Memory Addressing
##### NUMA nodes per socket = NPS1
##### ACPI SRAT L3 Cche As NUMA Domain = Disabled

### CPU Common Options
#### L1 Stream HW Prefetcher = Enable
#### L1 Stride Prefetcher = Auto
#### L1 Region Prefetcher = Auto
#### L2 Stream HW Prefetcher = Enable
#### L2 Up/Down Prefetcher = Auto

# Management Firmware Settings

Out-of-the-box.

# Fan Settings (8,100 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>100</b> 0xFF
 0a 3c 00
</pre>

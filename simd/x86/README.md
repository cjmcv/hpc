# Intel intrinsic instructions
Intel intrinsic instructions, which are C style functions that provide access to many Intel instructions - including Intel® SSE, AVX, AVX-512, and more - without the need to write assembly code.
* Intel’s Instruction Set Architecture (ISA) Extensions[link](https://software.intel.com/en-us/isa-extensions)
* Intel Intrinsics Guide - [link](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)

---

# Check which instruction sets the CPU supports

```bash
# Check the flags to see if there are any instruction sets you want to use.
$ cat /proc/cpuinfo 
# If you can't find the instruction set you want to use, but the chip clearly supports it, 
# then it may be related to the Linux kernel version. 
# For example, the Linux kernel version required for Intel AMX instructions is 5.16 or higher.
# https://www.phoronix.com/news/Intel-AMX-Lands-In-Linux
# Check the version of linux kernel. 
$ uname -r
```
/*
 * Copyright © 2023 Moffett System Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "numa.h"

#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <linux/ioctl.h>

namespace moffett {
namespace spu_backend {

#define MOFFETT_MMAP_NODE "/dev/mf-remap-pfn"
#define MOFFETT_MMAP_NODE1 "/dev/mf-remap-node-pfn"

#define MOFFETT_MMAP_IOC_MAGIC  	'k'
#define MOFFETT_MMAP_NODE_IOC_MAGIC 'o'

#define MOFFETT_MMAP_IOCFREE		_IOWR(MOFFETT_MMAP_IOC_MAGIC, 0x2, unsigned long)
#define MOFFETT_MMAP_IOCFREE1		_IOWR(MOFFETT_MMAP_NODE_IOC_MAGIC, 0x2, unsigned long)

typedef struct moffett_mmap_args
{
  uint64_t vaddr;
} moffett_mmap_args_t;


void* moffett_mmap2(size_t size, int numa)
{
  int fd;
  void *pvaddr = NULL;

  if (numa == 0) {
    fd = open(MOFFETT_MMAP_NODE, O_RDWR);
  } else {
    fd = open(MOFFETT_MMAP_NODE1, O_RDWR);
  }
  if (fd < 0) {
    return NULL;
  }

  pvaddr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, fd, 0);
  if (pvaddr == MAP_FAILED) {
    close(fd);
    return NULL;
  }

  close(fd);

  return pvaddr;
}

int moffett_munmap(void *pvaddr, size_t size, int numa)
{
  int ret = 0;
  int fd;
  moffett_mmap_args_t mmap_args;

#ifdef MFT_MMAP_DEBUG
  printf("%s, pvaddr:%p, size:%ld \n", __func__, pvaddr, size);
#endif

  if(!pvaddr || !size)
    return -1;

  if (numa == 0) {
    fd = open(MOFFETT_MMAP_NODE, O_RDWR);
  } else {
    fd = open(MOFFETT_MMAP_NODE1, O_RDWR);
  }
  if (fd < 0) {
    perror("/dev/mf-remap_pfn open failed \n");
    return -1;
  }

  munmap(pvaddr, size);

  memset(&mmap_args, 0, sizeof(moffett_mmap_args_t));
  mmap_args.vaddr = (uint64_t)pvaddr;
  if (numa == 0) {
    if (ioctl(fd, MOFFETT_MMAP_IOCFREE, &mmap_args) < 0) {
      ret = -1;
      printf("%s, Call cmd DMABUF_IOCFREE fail! \n", __func__);
    }
  } else {
    if (ioctl(fd, MOFFETT_MMAP_IOCFREE1, &mmap_args) < 0) {
      ret = -1;
      printf("%s, Call cmd DMABUF_IOCFREE fail! \n", __func__);
    }
  }

  close(fd);

  return ret;
}


void *MallocHostMemory(size_t size, int numa) {
  return moffett_mmap2(size, numa);
}

int FreeHostMemory(void* vaddr, size_t size, int numa) {
  return moffett_munmap(vaddr, size, numa);
}

}
}
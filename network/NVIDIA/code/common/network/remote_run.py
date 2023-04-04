# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import getpass
import paramiko

from typing import Tuple
from pathlib import Path
from code.common import logging


class RemoteConnection:
    """
    Open a remote connection via SSH and run command in the remote node
    NOTE: Shell can be invoked via invoke_ssh() but this usage is not as clean as exec_command()
    """

    def __init__(self,
                 server: str, port: int = 22,
                 user: str = getpass.getuser(),
                 private_key_file: str = str(Path('/mnt' + str(Path.home()), '.ssh', 'id_rsa'))):
        self.server = server
        self.port = port
        self.user = user
        self.private_key_file = Path(private_key_file).absolute()
        self.password = None
        self.client = None
        self._shell = None
        self._stdin = None
        self._stdout = None
        self._stderr = None

    def gather_password_from_user_input(self):
        while True:
            pw1 = getpass.getpass(f"Enter password for {self.user}: ")
            pw2 = getpass.getpass(f"Confirm password for {self.user}: ")
            if pw1 == pw2:
                self.password = pw1
                break
            else:
                logging.warning("Password you entered does not match, please try again...")

    """ first try key_based_connection, then password_based_connection """

    def connect(self):
        rtn = self.key_based_connection()
        if not rtn:
            self.gather_password_from_user_input()
            rtn = password_based_connection()
        assert rtn, "Failed to connect to {self.server}, terminating..."

    """ connect using private key """

    def key_based_connection(self) -> bool:
        self.paramiko_pkey = paramiko.RSAKey.from_private_key_file(self.private_key_file)
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.client.connect(self.server, self.port, username=self.user, pkey=self.paramiko_pkey)
        except Exception as e:
            logging.warning(f"Error: {e} while connecting to {self.user}@{self.server}:{self.port} via private key")
            logging.warning(f"Try ssh-copy-id user_id@server_ip_address later to set up passwordless ssh")
            return False
        logging.info(f"Connection to {self.server} established")
        return True

    """ connect using password """

    def password_based_connection(self) -> bool:
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.client.connect(self.server, self.port, username=self.user, password=self.password)
        except Exception as e:
            logging.warning(f"Error: {e} while connecting to {self.user}@{self.server}:{self.port} via password")
            return False
        logging.info(f"Connection to {self.server} established")
        return True

    """ run remote command using exec_command() """

    def run_remote_cmd(self, cmd: str):
        self._stdin, self._stdout, self._stderr = self.client.exec_command(cmd)

    """ receive remote command run outcome """

    def get_remote_cmd_rtn(self) -> Tuple[int, str, str]:
        out_exit = self._stdout.channel.recv_exit_status()
        out_stdout = self._stdout.read().decode()
        out_stderr = self._stderr.read().decode()
        return out_stdout, out_stderr, out_exit

    """ terminate remote connection """

    def close_remote_connection(self):
        self.client.close()
        logging.info(f"Connection to {self.server} closed")


# usage example
def main():
    server = "luna-prod-0586-80gb"
    MyRemConn = RemoteConnection(server)
    MyRemConn.key_based_connection()

    rem_cmd = " && ".join(["uname -a",
                           "pwd",
                           "ls -tlh",
                           "sleep 2",
                           "echo $PATH",
                           ])
    MyRemConn.run_remote_cmd(rem_cmd)
    print("run_remote_cmd() is nonblocking")
    print("......")
    print("......")
    stdout, stderr, rtnexit = MyRemConn.get_remote_cmd_rtn()
    print("rem_cmd exit  :\n", rtnexit)
    print("rem_cmd stdout:\n", stdout)
    print("rem_cmd stderr:\n", stderr)

    rem_cmd = " && ".join(["pwd",
                           "sleep 5",
                           "ls -tlh",
                           "/bin/false",
                           ])
    MyRemConn.run_remote_cmd(rem_cmd)
    stdout, stderr, rtnexit = MyRemConn.get_remote_cmd_rtn()
    print("rem_cmd exit  :\n", rtnexit)
    print("rem_cmd stdout:\n", stdout)
    print("rem_cmd stderr:\n", stderr)

    MyRemConn.close_remote_connection()

    MyRemConn = RemoteConnection(server)
    MyRemConn.gather_password_from_user_input()
    MyRemConn.password_based_connection()

    rem_cmd = " && ".join(["cd ~/",
                           "pwd",
                           "ls | head -n 10",
                           ])
    MyRemConn.run_remote_cmd(rem_cmd)

    stdout, stderr, rtnexit = MyRemConn.get_remote_cmd_rtn()
    print("rem_cmd exit  :\n", rtnexit)
    print("rem_cmd stdout:\n", stdout)
    print("rem_cmd stderr:\n", stderr)

    MyRemConn.close_remote_connection()


if __name__ == '__main__':
    main()

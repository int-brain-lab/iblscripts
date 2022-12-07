# Purpose of using ansible for server configurations

Ansible is a configuration management tool that attempts to utilize 'code' as documentation. With the assumption that 
documentation will always lag behind what is currently in use, this is a useful approach to keep them in sync. Server 
configurations are stored as yaml files, that are decently easy to read, and (for our use) those yaml files are stored in a 
github repository. The current intention is to schedule the servers to automatically pull the configurations every 15 minutes via 
a cron job and apply changes as needed.

The classical ansible deployment method is to use a single control node to push out configuration changes to many client 
machines. We will be reversing this by having many machines check in with a 'control node'. The 'control node' in our situation 
will be github. The configuration files are run via an ansible-pull command. To summarize, intended benefit of utilizing ansible 
for our server configurations:
* codify the server configurations
* turn our configuration files into documentation
* have the ability to quickly spin up a new server in case the current one fails (hardware failure or otherwise)
* know when and why changes occurred
* quickly being able to revert to a known good state

## Instructions to get a server to start pulling

Tested for Ubuntu 22.04 OS:
```bash
sudo apt install ansible
ansible-pull --url https://github.com/int-brain-lab/iblscripts.git deploy/ansible_playbooks/`server_name`/local.yml
```
* `server_name` is the name of the server that is currently being configured, i.e. `parede`
* run a `sudo crontab -l` command to verify that an ansible-pull cron job entry was created

If working on a test branch, simply add the checkout argument:
```bash
ansible-pull --checkout `branch_name` --url https://github.com/int-brain-lab/iblscripts.git deploy/ansible_playbooks/`server_name`/local.yml
```
i.e. something along the lines of 
`ansible-pull --checkout ansible_init --url https://github.com/int-brain-lab/iblscripts.git deploy/ansible_playbooks/alyx-dev/local.yml`

## How to develop an ansible playbook
When developing a playbook, it is often useful to use a vm or ec2 instance in conjunction with snapshots to quickly revert to a 
known state. It is also useful to create and modify a playbook locally prior to utilizing the `ansible-pull` command detailed 
above.  

To run a playbook locally, run a command like `ansible-playbook local_example.yml`

Content of the `local_example.yml` file
```yaml
---
  - name: "Local playbook example"
    hosts: localhost
    connection: local 
    tasks:

    - name: "executes an ls -lrt command"
      shell: "ls -lrt"
      register: "output"

    - debug: var=output.stdout_lines
```

## Ansible documentation:
* General:
  * https://docs.ansible.com/ansible/latest/cli/ansible-pull.html
* Modules:
  * https://docs.ansible.com/ansible/latest/collections/ansible/builtin/apt_module.html
  * https://docs.ansible.com/ansible/latest/collections/ansible/builtin/blockinfile_module.html
  * https://docs.ansible.com/ansible/latest/collections/ansible/builtin/cron_module.html
  * https://docs.ansible.com/ansible/latest/collections/ansible/builtin/file_module.html
  * https://docs.ansible.com/ansible/latest/collections/ansible/builtin/git_module.html
  * https://docs.ansible.com/ansible/latest/collections/ansible/builtin/lineinfile_module.html
  * https://docs.ansible.com/ansible/latest/collections/ansible/posix/mount_module.html
  * https://docs.ansible.com/ansible/latest/collections/ansible/builtin/shell_module.html
  * https://docs.ansible.com/ansible/latest/collections/ansible/builtin/user_module.html

#!/bin/bash
set -e

# Fix ownership/permissions on the bind-mounted project dir every start
chown -R racer:racer /home/racer/project
chmod -R 775 /home/racer/project

ssh-keygen -A
exec /usr/sbin/sshd -D
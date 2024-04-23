import paramiko
import guiHelper as gu

# SERVER ='della.princeton.edu'
#
# client = paramiko.SSHClient()
# client.load_system_host_keys()
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#
username = 'xinweiy'
# password = 'nanchang35768wztYXW'
# client.connect(SERVER, 22, username, password)client=gu.dellaConnect(username)

client=gu.dellaConnect(username)
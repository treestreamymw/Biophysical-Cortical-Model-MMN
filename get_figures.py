import paramiko
from scp import SCPClient


host='elnath.ni.tu-berlin.de'
user='gilikar'
keypath='/Users/gilikarni/.ssh/pem_id_rsa'


#k = paramiko.RSAKey.from_private_key_file(keypath)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(\
    paramiko.AutoAddPolicy())

client.load_system_host_keys()


client.connect(hostname=host,\
    username=user,key_filename=keypath)

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


gauth = GoogleAuth()           
drive = GoogleDrive(gauth)


upload_file_list = ['/home/duongnh/project_f19/photo_attack/result/b0_ver_3/train/checkpoint_b0_ver_3/cp_06.h5', 
                    '/home/duongnh/project_f19/photo_attack/result/b0_ver_1/train/checkpoint_b0_ver_1/cp_12.h5']
for upload_file in upload_file_list:
	gfile = drive.CreateFile({'parents': [{'id': '13B_agGVFZmWUECH6H3vQyz43rTw_6tvf'}]})
	# Read file and set it as the content of this instance.
	gfile.SetContentFile(upload_file)
	gfile.Upload() # Upload the file.








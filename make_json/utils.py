# Necessary packages
import os
# My packages

# <1>
def check_exist_folder(path):
    '''Check if the directory exists or not?
    Args:
      - path: path to directory
    Returns:
      - if directory is not exists, the new folder was created
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        print('not have directory save json, just made')
    else:
        print('already had directory save json')

def get_path(path, mode):
  list_paths = []
  count = 0
  for sub_img in os.listdir(path):
    sub_img_path = os.path.join(path, sub_img)
    live_path = os.path.join(sub_img_path, 'live')
    spoof_path = os.path.join(sub_img_path, 'spoof')

    temp_live = 'Data/'
    temp_spoof = 'Data/'

    if os.path.exists(live_path):
      for file in os.listdir(live_path):
        if '.png' in file or '.jpg' in file:
          a = 'Data' + '/' + mode + '/' + sub_img + '/' + 'live' + '/' + file
          list_paths.append(a)
          count += 1
          if count % 4000 == 0:
            print('load number of path:  ', count)

    if os.path.exists(spoof_path):
      for file in os.listdir(spoof_path):
        if '.png' in file or '.jpg' in file:
          b = 'Data' + '/' + mode + '/' + sub_img + '/' + 'spoof' + '/' + file
          list_paths.append(b)
          count += 1
          if count % 4000 == 0:
            print('load number of path:  ',count)

  return list_paths



def extract_sub_images(dictionary, sub_key, new_destination):
  # new destination is the path to folder in the computer
  # example new_destination = 'D:'
  sub_dictionary = {}
  for key, value in dictionary.items():
    if key in sub_key:
      new_key = new_destination +  key[4:]
      sub_dictionary.update({new_key: value})
  return sub_dictionary


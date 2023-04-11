from pywebcopy import save_webpage

url = 'https://www.svb.com/'
download_folder = './'

kwargs = {'bypass_robots': True, 'project_name': 'recognisable-name'}

save_webpage(url, download_folder, **kwargs)

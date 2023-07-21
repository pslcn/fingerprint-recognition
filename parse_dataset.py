import glob

def fingerprint_paths(path, finger, focus_id):
	return glob.glob(path + f"/*{finger[0].capitalize()}_{'_'.join([e for e in finger[1:]])}*.BMP")

def img_is_focus(img_path, focus_id):
	return int((img_path.split("/")[-1]).split("_")[0]) == focus_id

import os
import multiprocessing
from multiprocessing import Pool, current_process
import argparse
import subprocess
import time


import os

# f = open("2k_copyng_files.txt", "r")
# files_path = []
# for x in f:
#     #print(x)
#     # x = x[:-1]
#     # files_path.append(x.split(start_path)[1])
    
#     files_path.append(x[72:-1])
#     #print(files_path[-1])

    
# def search_file(path_glb):
#     for path in files_path:
#         if path.find(path_glb) != -1:
#             return path
#     return ""

def check_file4(path):
    #return False
    for i in range(4):
        path_img = os.path.join(path, f"{i:03d}.webp")
        if os.path.exists(path_img) == False:
            return False
        
    return True

def check_file36(path):
    #return False
    for i in range(4):
        path_img = os.path.join(path, f"4view_{i:03d}.webp")
        if os.path.exists(path_img) == False:
            return False
    for i in range(32):
       render_path = os.path.join(path, f"{i:03d}.webp")
       if os.path.exists(render_path) == False:
           return False
        
    return True

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    path_save
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break
        
        path_save_full = os.path.join(path_save, os.path.split(item)[1].split(".")[0])
        # if os.path.exists(view_path):
        #     queue.task_done()
        #     print('========', item, 'rendered', '========')
        #     continue
        # else:
        #     os.makedirs(view_path, exist_ok = True)

        # Perform some operation on the item
        print(item, gpu)
        os.makedirs(path_save_full, exist_ok = True)
    
        path_cameras_main = "/home/jovyan/3dgen/data/objaverse_mini_render/1_unpack"
        
        split_item = item.split("/")
        path_camera_item = os.path.join(path_cameras_main, split_item[-2], split_item[-1][:-4])
    
        path_cameras = ""#search_file(item.split("/")[-1])
        print(f"path_cameras: {path_cameras}")
        path_blender = "/home/jovyan/3dgen/users/nabiyarov/blender_render/objaverse-xl/scripts/rendering/"
        
        path_camera_full = ""#os.path.join(path_cameras_main, path_cameras[:-4])
        tmp_cameras = "./orig_cameras/8121ab069aa949fcb13811c1e4ab8a41"
        #if os.path.exists(path_camera_full) == False:
        path_camera_full = tmp_cameras
            
        
        
        for j in range(1):
            path_num_obj = os.path.join(path_save_full, str(j))
            command = (
                f" CUDA_VISIBLE_DEVICES={gpu} "
                f" blender-4.2.1-linux-x64/blender --background   --python blender_script_universal.py -- --object_path  {item} --num_renders 32 --output_dir {path_save_full} --path_cameras none --num_pbr 0 --light skybox --engine BLENDER_EEVEE_NEXT --number_view 4 --resolution 512 --format WEBP --normals=False --depth=False"
            )
            print(command)
            # if os.path.exists(os.path.join(path_save_full, "000.png"))==False or os.path.exists(os.path.join(path_save_full, "001.png")) == False or os.path.exists(os.path.join(path_save_full, "002.png")) == False or os.path.exists(os.path.join(path_save_full, "003.png"))==False:

            if check_file4(path_save_full) == False:
                #break
                print("No SKIP")
                time.sleep(3)
                #
                #
                subprocess.run(command, shell=True)
            else:
                print("SKIP")
        

        with count.get_lock():
            count.value += 1

        queue.task_done()


def start_multirender(num_gpus, cnt_in_gpu, path_with_glb, path_save_render):
    
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    
    for gpu_i in range(num_gpus):
        for worker_i in range(cnt_in_gpu):
            worker_i = gpu_i * cnt_in_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, path_save_render)
            )
            process.daemon = True
            process.start()

    for obj in os.listdir(path_with_glb):
        full_path_obj = os.path.join(path_with_glb, obj)
        queue.put(full_path_obj)    
     
    # with open('file_full_path.txt', 'r', encoding='utf-8') as f:
    #      for line in f:
    #         #print(line)
    #         #glbs.append(line[:-1])
    #         queue.put(line[:-1])
        
        
    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(num_gpus * cnt_in_gpu):
        queue.put(None)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--num_gpu',
    type=int,
    default=7
)
    parser.add_argument(
    '--path_with_glb',
    type=str)
    parser.add_argument(
    '--path_save',
    type=str)
    my_namespace = parser.parse_args()


    start_multirender(my_namespace.num_gpu, 2, my_namespace.path_with_glb, my_namespace.path_save)

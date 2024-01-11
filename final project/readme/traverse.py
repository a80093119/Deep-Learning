
import os

if __name__ == '__main__':

    THE_FOLDER = "./"

    for the_dir in os.listdir(THE_FOLDER):
        
        if not os.path.isdir(the_dir):
            continue

        json_path = THE_FOLDER+ the_dir+ f"/{the_dir}_feature.json"
        gt_path= THE_FOLDER+ the_dir+ "/"+ the_dir+ "_groundtruth.txt"

        youtube_link_path= THE_FOLDER+ the_dir+ "/"+ the_dir+ "_link.txt"

        print ("Data No:", the_dir)
        print ("---gt data path:", gt_path)
        print ("---Pitch data path:", json_path)
        print ("---Youtube link path:", youtube_link_path)

        try:
            with open(youtube_link_path, 'r') as yt_link:
                link= yt_link.readline()
                print ("------Youtube link:", link)
        except:
            print ("------Error: YT link file not exist or can't be read")


        try:
            with open(json_path, 'r') as json_file:
                print ("------Great, feature file can be opened")
        except:
            print ("------Error: feature file not exist or can't be read")

        try:
            with open(gt_path, 'r') as gt_file:
                print ("------Great, gt file can be opened")
        except:
            print ("------Error: gt file not exist or can't be read")

        print ("")
import NewPrograms.MainTrainer as MainTrainer
import CreateDataFile
import time

if __name__ == "__main__":
    start_time = time.time()
    
    #CreateDataFile = CreateDataFile.CreateDataFile()
    #CreateDataFile.execute()
    
    maintrainer = MainTrainer.MainTrainer()
    maintrainer.run()
    
    process_time = time.time() - start_time
    print("Process time: ", round(process_time, 0), " seconds")
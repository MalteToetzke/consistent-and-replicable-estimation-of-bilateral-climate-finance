import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

"""Import"""
# Define path
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

print(dir_path)

df = pd.read_csv(dir_path+'/Data/User_study/results.csv')

df.case[((df.ClimateAdaptation.isin([1, 2])) | (df.ClimateMitigation.isin([1, 2])))&(df.meta_category.isin(["Mitigation","Adaptation"]))]=1
df.case[((df.ClimateAdaptation.isin([1, 2])) | (df.ClimateMitigation.isin([1, 2])))&(df.meta_category.isin(["Environment"]))]=2
df.case[((df.ClimateAdaptation.isin([1, 2])) | (df.ClimateMitigation.isin([1, 2])))&(df.climate_relevance==0)]=3
df.case[((df.ClimateAdaptation.isin([0])) & (df.ClimateMitigation.isin([0])))&(df.meta_category.isin(["Mitigation","Adaptation"]))]=4

df["overlap"] = np.nan

df_response = df[["Student1","Student2","Student3","Student4","Student5","Student6","Student7","Student8","Student9","Student10"
                  ,"Student11","Student12","Student13","Student14","Student15","Student16","Student17","Student18"]]
for i,row in df_response.iterrows():
    agreement= 0
    disagreement = 0
    valid_responses = 0
    list_unique_response = []
    for response in row:
        if response in ["Adaptation","Mitigation","Environment","None"]:
            if response in ["Adaptation","Mitigation"]:
                response = True
            else:
                response = False
            valid_responses += 1
            if response in list_unique_response:
                agreement +=1
            else:
                if len(list_unique_response)>0:
                    disagreement +=1
                list_unique_response.append(response)
    try:
        if agreement!=0:
            overlap = (agreement+1)/(valid_responses)
        else:
            overlap=0
        df.at[i, "overlap"] = overlap
    except:
        print("max one valid response")

print(df.overlap)



fig = plt.figure()
mean_averages=[]
for i in [1,2,3,4]:
    ax = fig.add_subplot(4,1,i)

    df_case = df[df.case==i]
    confidence = np.mean(list(df_case.overlap[df_case.overlap.isnull()==False]))
    j=0
    averages= []
    for respondent in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:

        column_name = "Student"+str(respondent)
        print(df_case[df_case[column_name].isin(['Adaptation','Mitigation'])].shape[0])
        print(df_case[df_case[column_name].isin(['Adaptation','Mitigation','None','Environment'])].shape[0])
        try:
            average_evaluation = df_case[df_case[column_name].isin(['Adaptation','Mitigation'])].shape[0]/df_case[df_case[column_name].isin(['Adaptation','Mitigation','None','Environment'])].shape[0]
            print(average_evaluation)

            average_evaluation=round(average_evaluation,ndigits=3)
            averages.append(average_evaluation)
            ax.plot(average_evaluation,j, marker="v")
            #ax.text(average_evaluation,j,s= str(average_evaluation))
        except:
            pass

        plt.xlim(0,1)
        j+=0.01
    # reomove y axis
    ax.vlines(x=np.mean(averages), ymin=0, ymax=j, linestyle='--', color='black')##636363
    ax.text(x=np.mean(averages), y=j+0.02, s= str(round(np.mean(averages),2)),ha="center")
    mean_averages.append(np.mean(averages))
    ax.set(yticklabels=[])  # remove the tick labels
    if i<4:
        ax.set(xticklabels=[])
    ax.tick_params(left=False)
    ax.text(-0.15, j/2, s= "Case "+str(i))#,h="ha"
    #ax.text(1.02, j/2, s= "Mean overlap = "+str(round(confidence,2)))

# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    #right=0.9,
                    #top=0.9,
                    wspace=0.4,
                    hspace=0.4)

plt.savefig(dir_path+'/Data/User_study/Results/results_students.pdf', bbox_inches ="tight", dpi=1200)
plt.savefig(dir_path+'/Data/User_study/Results/results_students.png', bbox_inches ="tight", dpi=1200)
print(mean_averages)
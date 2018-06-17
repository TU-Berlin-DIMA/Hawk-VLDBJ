from exploration import * 

feature_wise_csvdata = concatCSVFiles(sys.argv[1])
feature_wise_data = createPandasDataFrame(feature_wise_csvdata)
feature_wise_output = processDataFeatureWise(feature_wise_data)

devices = ["phi"]

latex_table = ""

latex_table += "\\begin{table*}\n"
latex_table += "\\begin{center}\n"

latex_table += "\\begin{tabular}{ | c | " + "c | " * len(devices)
latex_table += "}\n"
latex_table += "\\hline\n"

latex_table += "Query & "
device_counter = 0
for device_type in devices:
    latex_table += device_type
    if device_counter < len(devices) - 1:
        latex_table += " & "
        
    device_counter += 1

latex_table += "\\\\\n"

latex_table += "\\hline\n"


for query in sorted(feature_wise_output.Query.unique()):
    latex_table += query.replace("_", "\\_") + " & "

    device_counter = 0    
    for device in devices:
        print(device + " " + query)
        filtered = feature_wise_output[(feature_wise_output["DeviceType"] == device) & (feature_wise_output["Query"] == query)]
        #print(filtered)
        latex_table += "{:10.3f}".format(filtered["Mean"].tolist()[0])
        
        if device_counter < len(devices) - 1:
            latex_table += " & "
        
        device_counter += 1
        
    latex_table += "\\\\\n"

latex_table += "\\hline\n"    
latex_table += "\\end{tabular}\n"
latex_table += "\\end{center}\n"
latex_table += "\\end{table*}\n"
    
print(latex_table)

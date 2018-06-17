from exploration import *

if __name__ == '__main__':
    scriptlocation = os.path.dirname(os.path.realpath(__file__))
    eprint("Scanning logs in directory:", sys.argv[1])
    with open(os.path.join(scriptlocation, 'report.tpl.tex.part1'), 'r') as tplfile:
        tplPart1=tplfile.read()
    with open(os.path.join(scriptlocation, 'report.tpl.tex.part2'), 'r') as tplfile:
        tplPart2=tplfile.read()
    with open(os.path.join(scriptlocation, 'report.tpl.tex.part3'), 'r') as tplfile:
        tplPart3=tplfile.read()
    csvdata = concatCSVFiles(sys.argv[1])
    data = createPandasDataFrame(csvdata)
    output = processData(data)

    feature_wise_csvdata = concatCSVFiles(sys.argv[2])
    feature_wise_data = createPandasDataFrame(feature_wise_csvdata)
    feature_wise_output = processDataFeatureWise(feature_wise_data)

    output = pd.concat([output, feature_wise_output])

    for query in output.Query.unique():

        result = ""

        for device in output.Device.unique():
            for varianttag in output.VariantTag.unique():
                filtered = output[(output["Device"] == device) & (output["Query"] == query) &
                                  (output["VariantTag"] == varianttag)]

                if filtered.empty:
                    continue

                result += "\\begin{filecontents*}{" + filtered["DeviceType"].unique()[0] + "_processor_"
                result += varianttag.replace(" ", "_") + "." + query + ".report.csv}\n"

                result += "id;DeviceType;Device;VariantTag;Variant;Min;Max;Median;Mean;Stdev;Var\n"

                result += "0;"

                cols = ["DeviceType", "Device", "VariantTag", "Variant", "Min", "Max", "Median", "Mean", "Stdev",
                        "Var"]
                result += filtered.to_csv(columns=cols, sep=';', index=False, float_format='%.9f', header=False)

                result += "\end{filecontents*}\n"

        with open(query + ".csv.tex", 'w') as file:
            file.write(result)

    for host in output.Host.unique():
        host_name = host.replace("_", "-")
        for query in output.Query.unique():
            query_name = query.replace("_", "-")
            eprint(host_name, query_name)
            queryoutput = output[(output.Host == host) & (output.Query == query)]
            cols = ["DeviceType", "Device", "VariantTag", "Variant", "Min", "Max", "Median", "Mean", "Stdev", "Var"]
            outputcsv = queryoutput.to_csv(columns=cols, sep=';', index=False, float_format='%.9f')
            outputcsv = outputcsv.replace("_", "-")

            with open(host + "." + query + ".report.tex", 'w') as reportfile:
                reportfile.write(tplPart1)
                reportfile.write(outputcsv)
                reportfile.write(tplPart2)
                reportfile.write(query_name + " on " + host_name)
                reportfile.write(tplPart3)

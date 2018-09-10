import matplotlib.pyplot as plt
# matplotlib inline

f = plt.figure()
plt.errorbar(
    [10,50,100,150,200,250,300],  # X
    [0.943354,0.969118,0.972498,0.970398,0.970672,0.972772,0.9717925], # Y
    yerr=[0.009424122,0.001724433,0.000878334,0.000677916,0.001005803,0.000945923,0.001682922],     # Y-errors
    label="Text GCN",
    fmt="ro-", # format line like for plot()
    linewidth=2	# width of plot line
    )


plt.legend() #Show legend
plt.show()

f.savefig("results/dim_R8.pdf", bbox_inches='tight')

f = plt.figure()
plt.errorbar(
    [10,50,100,150,200,250,300],  # X
    [0.75836,0.761456,0.76517,0.764888,0.767395,0.765452,0.764722], # Y
    yerr=[0.006711505,0.003761334,0.001671751,0.001010703,0.001961939,0.001658545,0.00168127],     # Y-errors
    label="Text GCN",
    fmt="ro-", # format line like for plot()
    linewidth=2	# width of plot line
    )


plt.legend() #Show legend
plt.show()

f.savefig("results/dim_MR.pdf", bbox_inches='tight')
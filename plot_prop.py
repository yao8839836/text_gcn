import matplotlib.pyplot as plt
# matplotlib inline

f = plt.figure()
plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.497666,0.707849,0.784678,0.806268], # Y
    yerr=[0.004616994,0.002982208,0.001491299,0.002531259],     # Y-errors
    label="Text GCN",
    fmt="r<--", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.149283108,0.334997347,0.485854698,0.752257032], # Y
    yerr=[0.016322364,0.029301421,0.046951708,0.008422076],     # Y-errors
    label="CNN-non-static",
    fmt="bo-", # format line like for plot()
    linewidth=2	# width of plot line
    )
plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.0955,0.30832,0.383746978,0.54306], # Y
    yerr=[0.049771506,0.005544527,0.040551828,0.010268544],     # Y-errors
    label="LSTM (pretrain)",
    fmt="g.--", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.302443002,0.569215862,0.669078598,0.732873], # Y
    yerr=[0.012816293,0.005403935,0.011636367,0.006793104],     # Y-errors
    label="Graph-CNN-C",
    fmt="yx-.", # format line like for plot()
    linewidth=2# width of plot line
    )
plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.114046734,0.451540096,0.660382369,0.746813595], # Y
    yerr=[0,0,0,0],     # Y-errors
    label="TF-IDF + LR",
    fmt="c>-.", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.legend() #Show legend
plt.show()

f.savefig("results/proportion_20ng.pdf", bbox_inches='tight')


f = plt.figure()
plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.883005,0.942432,0.943446,0.960034], # Y
    yerr=[0.00273641,0.00244692,0.001937984,0.001314206],     # Y-errors
    label="Text GCN",
    fmt="r<--", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.797259028,0.886614875,0.882960261,0.918730005], # Y
    yerr=[0.016831613,0.021104224,0.014498768,0.011398275],     # Y-errors
    label="CNN-non-static",
    fmt="bo-", # format line like for plot()
    linewidth=2	# width of plot line
    )
plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.192933333,0.89098,0.89780746,0.9309619], # Y
    yerr=[0.012816293,0.005403935,0.011636367,0.006793104],  # Y-errors
    label="LSTM (pretrain)",
    fmt="g.--", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.820557356,0.920922796,0.924257652,0.95185016], # Y
    yerr=[0.008723152,0.003612839,0.003155102,0.001634403],     # Y-errors
    label="Graph-CNN-C",
    fmt="yx-.", # format line like for plot()
    linewidth=2	# width of plot line
    )
plt.errorbar(
    [0.01,0.05,0.1,0.2],  # X
    [0.526724532,0.807217908,0.814527181,0.871630882], # Y
    yerr=[0,0,0,0],     # Y-errors
    label="TF-IDF + LR",
    fmt="c>-.", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.legend() #Show legend
plt.show()

f.savefig("results/proportion_R8.pdf", bbox_inches='tight')
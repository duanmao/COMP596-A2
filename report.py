import os
import glob

s = "<html><body style='width: 1200;'>"
s += "<p style=\"font-size: large; font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif; padding: 20;\">COMP596 Assignment 2 - Junlin Zheng 260833259 </p>"
s += "<ol type='a'>"
qs = ['Performance Comparison', 'Cluster Plots']
benchmarks = ['real-classic', 'real-node-label', 'LFR Benchmark (synthetic)']
tablewidths = [1000, 850, 1000, 750]
networks = {
    'real-classic': ["polbooks", "polblogs", "karate", "football", "strike"], 
    'real-node-label': ["citeseer", "cora", "pubmed"],
    'LFR Benchmark (synthetic)': [
        'LFR (n = 200, tau1 = 2.5, tau2 = 1.5, mu = 0.1, mindegree = 5, seed = 10)',
        'LFR (n = 200, tau1 = 2.5, tau2 = 1.5, mu = 0.1, mindegree = 5, seed = 11)',
        'LFR (n = 211, tau1 = 2.5, tau2 = 1.5, mu = 0.1, mindegree = 5, seed = 10)',
        'LFR (n = 217, tau1 = 2.5, tau2 = 1.5, mu = 0.1, mindegree = 3, seed = 11)',
        'LFR (n = 222, tau1 = 2.5, tau2 = 1.5, mu = 0.1, mindegree = 6, seed = 10)'
    ]
}
methods = ["Louvain", "Greedy Modularity", "Infomap", "Walk Trap", "Label Propagation", "Leading Eigenvector", "Edge Betweenness", "Spinglass"]

for i, q in enumerate(qs):
    s += f'<li style="font-size: larger; font-family: fantasy; margin-bottom: 10pt;">{q}</li>'
    if i == 0:
        s += '<p style="font-size: large; font-family: monospace;">'
        s += 'Below are tables listing each NMI, ARI and modularity of each method used to cluster each network of the different benchmarks, and the overall performance comes the last.</br>'
        s += 'For each type of the benchmarks and the overall average, the best approach that achieves the highest performance is marked by the blue border in the table.'
        s += '</p>'
        
        s += '<ul>'
        benchmarks.append('all')
        for j, benchmark in enumerate(benchmarks):
            s += f'<li style="font-size: larger; font-family: fantasy;">{benchmark}</li>'
            tablepath = f'tables/{benchmark}.png'
            s += f'<img src="{tablepath}" width="{tablewidths[j]}"/>'
            if benchmark == 'real-classic':
                s += '<p style="font-family: monospace;">* </br> Although it seems that Spinglass and Edge Betweenness beat other approaches with their higher average performace in terms of NMI and/or ARI, they are not chosen as the best performance winner. The reason for this is that the results of them on the largest network "polblogs" were not successfully reported, which turned out to be the lowest results among all the networks and should have a significant impact on the average value. They did not finish the work due to different reasons: Spinglass encounter does not work on unconnected graph and Edge Betweenness sometimes takes too much time to complete. </br> Given this, I prefer to choose a method that covered all networks as the winner. </br> Same logic adopted when comparing overall performance below.'
            s += '<p></br></p>'
        s += "</ul>"
        benchmarks.pop()
    else:
        s += '<ul>'
        for benchmark in benchmarks:
            s += f'<li style="font-size: larger; font-family: fantasy;">{benchmark}</li>'

            s += '<ul>'
            for i, network in enumerate(networks[benchmark]):
                s += f'<li style="font-size: large; font-family: fantasy;">{network}</li>'

                s += '<ul>'
                for method in methods:
                    title_method = method
                    if method == "Greedy Modularity":
                        title_method = "Fast Modularity"
                    title_network = network
                    if benchmark == 'LFR Benchmark (synthetic)':
                        title_network = i
                    plotpath = f'plots/{benchmark} {title_network} {title_method}.png'
                    
                    if os.path.isfile(plotpath):
                        s += f'<li style="font-family: fantasy;">{method}</li>'
                        s += f'<img src="{plotpath}" />'

                s += '</ul>'

            s += '</ul>'
        s += '</ul>'

s += "</ol>"
s += "</body></html>"
f = open("report.html", 'w')
f.write(s)
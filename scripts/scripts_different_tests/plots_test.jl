using Plots
x1 = 1:10
x2 = 1:2:20

y1 = rand(10)
y2 = rand(10)
y3 = rand(10)
y4 = rand(10)

using Plots

# Plot first group
# Add ungrouped entry
plot(x1, y3, color=:red, linewidth=2, label="lab1")
plot!(x1, y4, color=:green, linewidth=2, label="lab2")
plot!([], [], label="lab3", color=:blue, linewidth=2, opacity=0.0)




# Add second group member
# plot!(x1, y4, color=:green, linewidth=2, group=3)

# Customize legend with manual grouping
# plot!(legend=:topright, legendtitle="Series Groups")

savefig("grouped_plot_example.png")
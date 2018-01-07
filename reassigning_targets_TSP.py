import showdesigntool as sdt
import numpy as np
import optimization4doc as o4d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('seaborn-notebook')



input_folder = "~/tmp/..."
showElementNameSeq = [
                      '04_M_sinus.DAE',
                      '05_M_guitar_lightningstrikes.DAE'
                       ]
name = "Sinus-Guitar-Reassign-Targets"
fixed_end = False
individual_m2m_reassigning = True
C = 6.0
vmax = 2.0
framerate = 1
showElementSeq = [sdt.cld.loadShowElement(input_folder, name) for name in showElementNameSeq]
loc = "lower left"


# Simple case of parallel lines (overrides meshes)
# N = 20
# d0 = np.zeros((N, 3))
# d0[:, 1] = 100*np.arange(N)
# d1 = np.zeros((N, 3))
# d1[:, 1] = 100*np.arange(N)
# showElementSeq[0] = d0
# showElementSeq[1] = d1
# loc = "center right"


n = showElementSeq[0].shape[0]
showElementSeq[1][:, 2] += 150
m0 = showElementSeq[0]
m1 = showElementSeq[1]


# Create simanneal instance
ind0 = np.arange(n)
ind1 = np.arange(n)
np.random.shuffle(ind1)
initial_state = list(ind0) + list(ind1)
show = o4d.WholeShow(showElementSeq, initial_state, fixed_end, vmax)
#auto_schedule = show.auto(5.)
schedule = {'updates': 100,  # does not affect results
            'tmin': .00001,
            'tmax': 10000.0,
            'steps': 1000000
            }
show.set_schedule(schedule)
show.copy_strategy = "slice"
final_state, final_E, EList, stateList = show.anneal()


# Create meshList with optimally ordered
showElementSeqOpt = []
animationsDuration = 0
for m, el in enumerate(showElementSeq):
    showElementSeqOpt.append(el[final_state[m * n:(m + 1) * n], :])


print("Duration initial configuration: {} s".format(EList[0]))
print("Duration optimized configuration: {} s".format(EList[-1]))




# Data preparation

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return list(ret[n - 1:] / n)


# Smoothen & Subsample
Nsub = int(len(EList)/60)
E = moving_average(EList, int(Nsub/5))
iterations = [i for i in range(len(E))]
iterations = iterations[::Nsub]
E = E[::Nsub]
stateList = stateList[::Nsub]
stateList = stateList[:-1]

# Append final values again - in case they got lost by subsampling
E.append(final_E)
stateList.append(final_state)
iterations.append(iterations[-1]+(iterations[1]-iterations[0]))

# Find where energy minimum has been reached
end = np.where(E == min(E))[0][0]
E = E[0:end]
stateList = stateList[0:end]
iterations = iterations[0:end]

# Append final state
Nappend = int(2e-1*len(E))
for i in range(Nappend):
    E.append(final_E)
    stateList.append(final_state)
    iterations.append(iterations[-1]+(iterations[1]-iterations[0]))

# Scale x-axis: iterations
iterations = [el/1000 for el in iterations]



# PLOT 1: Animation of simulated annealing process

fs = 20
ms = 10

def animate(t):
    """
    Updates data in the plot.

    """

    # Energy
    line1.set_xdata(iterations[0:t])
    line1.set_ydata(E[0:t])

    # Meshes & Trajectories
    ax2.clear()
    target, = ax2.plot(m1[:, 1], m1[:, 2],
                       'o',
                       color='c',
                       markersize=ms
                       )
    source, = ax2.plot(m0[:, 1], m0[:, 2],
                       'o',
                       color='orange',
                       markersize=ms
                       )
    # Index labels
    source_ids = []
    target_ids = []
    for i in range(n):
        source_id = ax2.annotate(str(stateList[0][i]), (m0[stateList[0][i], 1], m0[stateList[0][i], 2]), fontsize=fs)
        target_id = ax2.annotate(str(stateList[t][i]), (m1[stateList[t][i+n], 1], m1[stateList[t][i+n], 2]), fontsize=fs)
        source_ids.append(source_id)
        target_ids.append(target_id)

    # Trajectories
    arrows = []
    for i in range(n):
        arrow = ax2.annotate("", xy=(m1[stateList[t][i+n], 1], m1[stateList[t][i+n], 2]),
                            xytext=(m0[stateList[0][i], 1], m0[stateList[0][i], 2]),
                            arrowprops=dict(width=2, headlength=25, headwidth=10),
                            color='k',
                            alpha=0.33
                            )
        arrows.append(arrow)

    return line1, source, target, source_ids, target_ids, arrows


def init():
    """
        Initialization of the plot.
    """

    # Energy
    line1.set_xdata(iterations)
    line1.set_ydata(E)

    # Meshes
    ax2.clear()
    target, = ax2.plot(m1[stateList[0][n::], 1], m1[stateList[0][n::], 2],
                       'o',
                       color='c',
                       label='Target Mesh Vertices',
                       markersize = ms
                       )
    source, = ax2.plot(m0[stateList[0][0:n], 1], m0[stateList[0][0:n], 2],
                       'o',
                       color='orange',
                       label='Source Mesh Vertices',
                       markersize=ms
                       )

    # Index labels
    source_ids = []
    target_ids = []
    for i in range(n):
        source_id = ax2.annotate(str(stateList[0][i]), (m0[stateList[0][i], 1], m0[stateList[0][i], 2]), fontsize=fs)
        target_id = ax2.annotate(str(stateList[0][i]), (m1[stateList[0][i+n], 1], m1[stateList[0][i+n], 2]), fontsize=fs)
        source_ids.append(source_id)
        target_ids.append(target_id)

    # Trajectories
    arrows = []
    for i in range(n):
        arrow = ax2.annotate("", xy=(m1[stateList[0][i+n], 1], m1[stateList[0][i+n], 2]),
                            xytext=(m0[stateList[0][i], 1], m0[stateList[0][i], 2]),
                            arrowprops=dict(width=2, headlength=25, headwidth=10),
                            color='k',
                            alpha=0.33
                            )
        arrows.append(arrow)

    return line1, source, target, source_ids, target_ids, arrows


fig, ax1 = plt.subplots()

# Energy
line1, = ax1.plot(iterations, E,
                  color='m',
                  label='Duration of Transition',
                  linewidth=5.0
                  )
ax1.set_xlabel("K Iterations", fontsize=fs)
dE = 5e-2*(max(E)-min(E))
ax1.set_ylim(min(E)-dE, max(E)+dE)
ax1.set_ylabel("Duration of Transition in s", fontsize=fs)
ax1.tick_params(
    axis='both',
    labelsize=fs
)

# Meshes & Trajectories
ax20 = ax1.twinx()
ax20.tick_params(
    axis='both',
    labelright='off',
    right='off',
    )
ax2 = ax20.twiny()
target, = ax2.plot(m1[stateList[0][n::], 1], m1[stateList[0][n::], 2],
                   'o',
                   color='c',
                   label='Target Mesh Vertices',
                   markersize=ms
                   )
source, = ax2.plot(m0[stateList[0][0:n], 1], m0[stateList[0][0:n], 2],
                   'o',
                   color='orange',
                   label='Source Mesh Vertices',
                   markersize=ms
                   )
ax2.tick_params(
    axis='both',
    labeltop='off',
    top='off',
    )

# All legends in one
lns = [line1, target, source]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=loc, fontsize=fs)

# Animation function
ani = animation.FuncAnimation(fig,
                              animate,
                              init_func=init,
                              interval=1.0,  # 1 ms between frames
                              frames=len(E),
                              repeat=True,
                              repeat_delay=2000,  # 2 seconds of delay before repeat
                              blit=False
                              )

# Window layout
plt.tight_layout()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.draw()
plt.show(ani)




# PLOT 2: Optimal configuration

def plot_configuration(t, stateList, EList, m0, m1, name, save=True):

    if t not in [0, -1]:
        print("Cannot plot this configuration.")
        return

    fig2, ax = plt.subplots()

    # Meshes
    target, = ax.plot(m1[stateList[t][n::], 1], m1[stateList[t][n::], 2],
                      'o',
                      color='c',
                      label='Target Mesh Vertices',
                      markersize=ms
                      )
    source, = ax.plot(m0[stateList[0][0:n], 1], m0[stateList[0][0:n], 2],
                      'o',
                      color='orange',
                      label='Source Mesh Vertices',
                      markersize=ms
                      )
    ax.tick_params(
        axis='both',
        labelleft='off',
        left='off',
        labelbottom='off',
        bottom='off'
    )

    ax.legend(loc=loc, fontsize=fs)

    # Index labels
    for i in range(n):
        ax.annotate(str(stateList[0][i]), (m0[stateList[0][i], 1], m0[stateList[0][i], 2]), fontsize=fs)
        ax.annotate(str(stateList[t][i]), (m1[stateList[t][i+n], 1], m1[stateList[t][i+n], 2]), fontsize=fs)

    # Trajectories
    arrows = []
    for i in range(n):
        arrow = ax.annotate("", xy=(m1[stateList[t][i+n], 1], m1[stateList[t][i+n], 2]),
                            xytext=(m0[stateList[0][i], 1], m0[stateList[0][i], 2]),
                            arrowprops=dict(width=2, headlength=25, headwidth=10),
                            color='k',
                            alpha=0.33
                            )
        arrows.append(arrow)


    # Set title
    if t == -1:
        title = 'Optimized configuration, transition takes {} s.'.format(EList[-1])
        n_suffix = "_opti.png"
    elif t == 0:
        title = 'Random configuration, duration takes {} s.'.format(EList[0])
        n_suffix = "_random.png"
    plt.title(title, fontsize=fs)

    # Window layout
    plt.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.draw()
    plt.show()

    if save:
        fig2.savefig(name + "{}-{}".format(EList[0], EList[-1]) + n_suffix)

    return


# Initial Configuration (t=0)
plot_configuration(0, stateList, EList, m0, m1, name, save=True)
# Optimal Configuration (t=-1)
plot_configuration(-1, stateList, EList, m0, m1, name, save=True)


print('Saving animation as GIF...')
ani.save(name + str(EList[0]) + "-" + str(EList[-1]) + '.gif', writer='imagemagick')

import matplotlib.pyplot as plt

def plot_joint_angles(angles_data):
    fig, ax = plt.subplots()
    ax.plot(range(len(angles_data)), angles_data, label="Joint Angle (°)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (°)")
    ax.set_title("関節角度の変化")
    ax.legend()
    return fig


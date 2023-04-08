import json

import matplotlib.pyplot as plt


def custom_analysis_eh_st_rr():
    with open("./grid_search_eh_st_rr") as f:
        lines = json.load(f)

    target_lines = {}
    for line in lines:
        if "eh2500" in line:
            if "st0.5" in line:
                target_lines[line] = lines[line]

    fig, ax = plt.subplots()
    for line in target_lines.values():
        ax.plot(line)
    plt.legend(target_lines.keys())
    # plt.savefig(f"./images/hyper_params_{target}.png")
    plt.show()
    # plt.close()


def analysis_eh_st_rr():
    with open("./grid_search_eh_st_rr") as f:
        lines = json.load(f)

    targets = ["eh2500", "eh5000"]
    for target in targets:
        target_lines = {}
        for line in lines:
            if target in line:
                target_lines[line] = lines[line]

        fig, ax = plt.subplots()
        for line in target_lines.values():
            ax.plot(line)
        plt.legend(target_lines.keys())
        plt.savefig(f"./images/hyper_params_{target}.png")
        plt.close()


def analysis_multi_race():
    with open("./grid_search_multi_race") as f:
        lines = json.load(f)

    fig, ax = plt.subplots()
    for line in lines.values():
        ax.plot(line)
    plt.legend(lines.keys())
    plt.savefig(f"./images/hyper_params_multi_race.png")
    plt.close()


if __name__ == '__main__':
    # analysis_eh_st_rr()
    # analysis_multi_race()
    custom_analysis_eh_st_rr()

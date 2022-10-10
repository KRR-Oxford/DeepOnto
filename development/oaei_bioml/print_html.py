import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from deeponto import SavedObj

import click

@click.command()
@click.option("-t", "--tool_name", type=str)
@click.option("-u", "--us_results", type=str)
@click.option("-s", "--ss_results", type=str)
def main(tool_name, us_results, ss_results):
    us = SavedObj.load_json(us_results)
    us_p = us["P"]
    us_r = us["R"]
    us_f1 = us["f_score"]
    ss = SavedObj.load_json(ss_results)
    ss_p = ss["P"]
    ss_r = ss["R"]
    ss_f1 = ss["f_score"]
    print(
        "  <tr>\n" +
        f"    <td>{tool_name}</td>\n" +
        f"    <td>{us_p}</td>\n" +
        f"    <td>{us_r}</td>\n" + 
        f"    <td>{us_f1}</td>\n" +
        f"    <td></td>\n" +
        f"    <td></td>\n" +
        f"    <td>{ss_p}</td>\n" +
        f"    <td>{ss_r}</td>\n" + 
        f"    <td>{ss_f1}</td>\n" +
        f"    <td></td>\n" +
        f"    <td></td>\n" +
        "  </tr>"
    )
main()
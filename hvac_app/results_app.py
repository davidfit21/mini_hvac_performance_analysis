import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import pdfkit
import numpy as np
import getpass
import base64
import platform
import subprocess

def generate_plot_points_table(cop_results):

    tables = {}
    for fluid, res in cop_results.items():
        df = pd.DataFrame({
            'OAT (°C)': res['X'].flatten(),
            'COP': res['y']  # these are the COP values used in the plot
        })
        tables[fluid] = df
    return tables


def generate_pdf_report_with_plot_points(plot_points_tables, plots_dict, filename="HVAC_Report.pdf"):
    # Get list of all fluids
    fluids_list = list(plot_points_tables.keys())
    title_fluids = ", ".join(fluids_list)
    
    # Determine output path
    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    output_pdf = os.path.join(downloads_dir, filename)

    # Start HTML content
    html_content = f"<html><head><title>HVAC Report</title></head><body>"
    html_content += f"<h1>HVAC Report for {title_fluids}</h1>"

    # Embed the combined plot at the top
    combined_plot_path = plots_dict.get("All Fluids", None)
    if combined_plot_path and os.path.exists(combined_plot_path):
        with open(combined_plot_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        html_content += f'<img src="data:image/png;base64,{img_base64}" width="800"><br><hr><br>'

    # Add tables for each fluid
    for fluid, table_df in plot_points_tables.items():
        html_content += f"<h2>{fluid}</h2>"
        html_content += "<table border='1' cellpadding='5'><tr><th>OAT</th><th>COP</th></tr>"
        for _, row in table_df.iterrows():
            html_content += f"<tr><td>{row['OAT (°C)']:.2f}</td><td>{row['COP']:.2f}</td></tr>"
        html_content += "</table><br>"

    html_content += "</body></html>"

    # Configure PDFKit
    config = pdfkit.configuration(
        wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    )
    pdfkit.from_string(html_content, output_pdf, configuration=config)

    # Open PDF cross-platform
    if platform.system() == "Windows":
        os.startfile(output_pdf)
    elif platform.system() == "Darwin":
        subprocess.run(["open", output_pdf])
    else:
        subprocess.run(["xdg-open", output_pdf])

    print(f"PDF report saved at {output_pdf}")
    return output_pdf
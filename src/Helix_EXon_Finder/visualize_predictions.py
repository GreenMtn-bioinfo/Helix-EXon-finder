### This script creates a self-contained HTML report with Plotly plots for each sequence analyzed by HEX-finder,
### so it should only be run once HEX-finder has been run and GFFs exist within the "../predictions/" directory.
### HEX-finder predictions can be visualized alone or, if local coordinates for reference exons are on-hand,
### predictions can be plotted alongside these features as a source of truth/qualitative performance evaluation. 
### Please see the preprint for a more quantitative performance evaluation (https://doi.org/10.64898/2025.12.19.694709).
### If you use the 'get_demo_seqs.py' tool to fetch sequences, it will automatically prepare the truth/reference features 
### in the correct format for use with this tool. As such, even if you plan on running HEX-finder on your own sequences AND benchmark features, 
### it is worth trying this out to see firsthand the appropriate format for the GFF file that this script can take as a truth source. 
### That said, visualization can be conducted without comparison of predictions to other features. Also, Options exist to skip plotting 
### sequences where the HEX-finder predictions and/or compared reference features show nothing if desired (see below).



# TODO: Plot customization?
# TODO: Add an additional profile visualization script that can be used if structural profiles are saved/available?



def main(args):
    
    from pathlib import Path
    from .paths import TEMP_DIR, FAVICON
    from .hex_finder import NO_PREDICTIONS_PATTERN, NO_PREDICTIONS_GROUPS
    from .utility_methods import import_gff, check_command_exit
    import re
    import base64
    import os
    import textwrap
    from bs4 import BeautifulSoup
    from bs4.formatter import HTMLFormatter
    import subprocess
    import shutil
    import plotly.graph_objects as go
    import plotly.io as pio
    from colorama import Fore, init
    init(autoreset=True)
    
    
    
    # Assign command-line arguments to Python variables
    truth_features_path = Path(args.truth_features) if args.truth_features else None
    predictions_dir = Path(args.predictions_dir)
    output_path = Path(args.output_path)
    truth_labels_field = args.truth_labels_attribute
    truth_source_name = args.truth_source_name
    quiet = args.verbose
    skip_empty = args.skip_empty
    include_javascript = args.javascript_included if args.javascript_included else 'cdn'
    accessibility_colors = args.accessibility_colors
    
    
    
    ### SCRIPT PATHS AND CONSTANTS (DO NOT CHANGE)
    temp_dir = TEMP_DIR

    # These are for formatting the CSS styles for divisions in the final report
    favicon_path = FAVICON
    shadow_color = 'rgba(0, 0, 0, 0.3)'
    shadow_intensity = '4px 4px 8px'
    plot_width_percent = "98%" # How much of the HTML page width does the plot occupy?
    body_background_color = '#e0e0e0'
    font_scaler = 1.2
    dropdown_font_size=int(14*font_scaler)
    seq_ID_color = "#2a496b" if accessibility_colors else '#175599'
    
    # Two color palette for the plot, the default 
    default_plotly_color_theme = dict(hf_color="royalBlue",
                                        hf_opacity=1,
                                        truth_color="black",
                                        truth_opacity=1,
                                        fp_color="red",
                                        fp_opacity=0.5,
                                        tp_color="green",
                                        tp_opacity=0.7,
                                        partial_color="orange",
                                        partial_line_thickness=1,
                                        not_covered_color="grey",
                                        not_covered_opacity=0.27)
    
    # And one that works better for those with most forms of color-blindness
    alt_plotly_color_theme = dict(hf_color= "#A020F0",
                                  hf_opacity=1,
                                  truth_color="#000000",
                                  truth_opacity=1,
                                  fp_color="#E69F00",
                                  fp_opacity=0.6,
                                  tp_color="#009E73",
                                  tp_opacity=0.8,
                                  partial_color="#009E73",
                                  partial_line_thickness=1.5,
                                  not_covered_color="#808080",
                                  not_covered_opacity=0.27)
    
    plotly_colors = alt_plotly_color_theme if accessibility_colors else default_plotly_color_theme
    
    
    
    ### FUNCTION DEFINITIONS
    
    def parse_gff(gff_path: str,
                  no_pred_expr: re.Pattern = NO_PREDICTIONS_PATTERN,
                  no_pred_groups: dict = NO_PREDICTIONS_GROUPS,
                  annot_delim: str = ";",
                  annot_assign: str = "=") -> list:
        """
        Imports and parses features from a GFF file given its path. 
        Handles GFFs created by HEX-finder slightly differently (looks for sequence length and confidence score attributes).
        """
        
        # Used in the loop below to retrieve/parse annotation from the attributes field of the GFF into a dictionary
        parse_attributes = lambda attributes: {attr.split(annot_assign)[0] : attr.split(annot_assign)[1] if len(attributes) > 1 else None for attr in attributes.split(annot_delim)}
        
        # Iterate through each line in from the GFF, handling the tab delimited fields (depending on whether they were generate by HEX-finder or not)
        feature_lines = import_gff(gff_path)
        exons = []
        if feature_lines:
            match = re.match(no_pred_expr, feature_lines[0][0])
            if match:
                seq_id = match.group(no_pred_groups['sequence_id'])
                seq_length = int(match.group(no_pred_groups['sequence_length']))
                return seq_id, None, None, seq_length, exons
            else:
                for i, line in enumerate(feature_lines):
                    annotation = parse_attributes(line[8])
                    if i == 0: # Certain information is pulled from the first line only (expected redundant)
                        seq_id, seq_source, feature_type = line[0:3]
                        seq_length = int(annotation['SEQUENCE_LENGTH']) if 'SEQUENCE_LENGTH' in annotation else None
                    score = float(annotation['CONFIDENCE_SCORE']) if 'CONFIDENCE_SCORE' in annotation else None
                    start = int(line[3])
                    end = int(line[4])
                    exons.append((start, end, score, annotation))
                return seq_id, seq_source, feature_type, seq_length, exons
        else:
            return None, None, None, None, exons


    def create_feature_lane_plot(feature_tuples,
                                seq_length,
                                seq_id,
                                lane_name="HEX-finder",
                                y_pos=1,
                                hf_color="royalBlue",
                                hf_opacity=1,
                                truth_color="black",
                                truth_opacity=1,
                                fp_color="red",
                                fp_opacity=0.5,
                                tp_color="green",
                                tp_opacity=0.7,
                                partial_color="orange",
                                partial_line_thickness=1.5,
                                not_covered_color="grey",
                                not_covered_opacity=0.35,
                                line_thickness=20,
                                truth_features: dict = None,
                                shade_margin: int = int(26/2 + 76/2),
                                # Customization Options
                                plot_title=None,
                                plot_height=400,
                                add_note=False,
                                xaxis_label="Position in sequence (nucleotide index)",
                                font_family="Arial",
                                title_font_size=int(24*font_scaler),
                                title_bold=True,
                                label_font_size=int(18*font_scaler),
                                label_bold=False,
                                tick_font_size=int(15*font_scaler),
                                tick_bold=False,
                                # Legend Options
                                legend_font_size=int(13*font_scaler),
                                margin_label="No predictions made at the ends",
                                match_label="Exact match",
                                no_match_label="False positive"):
        """
        Plots features in a DNA sequence as thick horizontal line segments (like a genome browser) using Plotly.
        Exon predictions (feature_tuples) have standard Plotly hover info.
        Truth features (truth_features) support custom labels (e.g. gene symbol from RefSeq) in hover info if provided.
        """
        
        fig = go.Figure()
        x_vals = []
        y_vals = []
        seq_bounds = (1, seq_length)
        
        
        # Prepare Y-axis tick text list
        current_tick_vals = [y_pos]
        current_tick_text = [lane_name]
        
        ## Process truth features if offered
        if truth_features is not None:
            truth_list = truth_features.get('features', [])
            
            # Prepare Match Sets (Coordinate Only)
            # We extract only (start, end) for the logic, ignoring the label if present
            truth_coords_only = []
            for item in truth_list:
                truth_coords_only.append((item[0], item[1]))
                
            truth_set = set(truth_coords_only)
            truth_set_flattened = {item : tup for tup in truth_set for item in tup}
            
            # Build Truth Trace with Custom Labels
            truth_x_vals = []
            truth_y_vals = []
            truth_custom_labels = [] # List for hover text
            
            default_label = truth_features.get('source', 'Reference')
            
            for item in truth_list:
                # Handle variable tuple length
                start = item[0]
                end = item[1]
                # If 3rd element exists, use it as label; otherwise use source name
                label = str(item[2]) if len(item) > 2 and item[2] is not None else default_label
                
                truth_x_vals.extend([start, end, None])
                truth_y_vals.extend([y_pos+1, y_pos+1, None])
                
                # Align label with geometry
                truth_custom_labels.extend([label, label, None])
            
            fig.add_trace(go.Scatter(
                x=truth_x_vals,
                y=truth_y_vals,
                mode='lines',
                line=dict(
                    color=truth_features.get('color', truth_color), 
                    width=line_thickness,
                ),
                name=default_label,
                # Apply custom labels here
                customdata=truth_custom_labels,
                hovertemplate='<b>%{customdata}</b><br>%{x:,}<extra></extra>',
                opacity=truth_opacity,
                showlegend=True
            ))
            
            # Add truth label to ticks
            current_tick_vals.append(y_pos + 1)
            current_tick_text.append(default_label)
        
        
        ## Process HEX-finder predictions (Standard tuples)
        if feature_tuples:
            for start, end in feature_tuples:
                x_vals.extend([start, end, None])
                y_vals.extend([y_pos, y_pos, None])
        
        # Add the single trace for all predictions
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(
                color=hf_color, 
                width=line_thickness
            ),
            name=lane_name,
            hovertemplate='%{x:,}',
            opacity=hf_opacity,
            showlegend=True
        ))
        
        
        # Check predictions against truth (to plot FP/TP later)
        exact_matches = [] 
        partial_matches = [] 
        partial_drawn = False
        no_matches = []
        
        if truth_features is not None:
            if feature_tuples:
                for feat in feature_tuples:
                    start, end = feat
                    found_match = False 
                    
                    # Check for exact matches
                    if tuple(feat) in truth_set:
                        exact_matches.append(feat)
                        found_match = True
                    
                    # Check for partial matches (start)
                    if start in truth_set_flattened.keys():
                        partial_matches.append(start)
                        found_match = True
                        
                    # Check for partial matches (end)
                    if end in truth_set_flattened.keys():
                        partial_matches.append(end)
                        found_match = True
                    
                    # If no exact or partial match was found, record it
                    if not found_match:
                        no_matches.append(feat)
        
        
        # Prepare axis and plot titles and ticks
        final_title_str = plot_title if plot_title is not None else f"Exon-level predictions for {seq_id}{' *' if add_note else ''}"
        final_title_text = f"<b>{final_title_str}</b>" if title_bold else final_title_str
        final_xaxis_text = f"<b>{xaxis_label}</b>" if label_bold else xaxis_label
        if tick_bold:
            final_y_tick_text = [f"<b>{t}</b>" for t in current_tick_text]
        else:
            final_y_tick_text = current_tick_text
        x_tick_prefix = "<b>" if tick_bold else ""
        x_tick_suffix = "</b>" if tick_bold else ""
        
        # Clean up layout
        fig.update_layout(
            title=dict(
                text=final_title_text,
                font=dict(family=font_family, size=title_font_size, color="black")
            ),
            xaxis_title=dict(
                text=final_xaxis_text,
                font=dict(family=font_family, size=label_font_size, color="black")
            ),
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='black',
                ticks='outside',
                tickcolor='black',
                zeroline=False,
                range=seq_bounds,
                tickfont=dict(family=font_family, size=tick_font_size, color="black"),
                tickprefix=x_tick_prefix, 
                ticksuffix=x_tick_suffix
            ),
            yaxis=dict(
                showgrid=True,
                showline=True,
                linecolor='black',
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=True,
                tickvals=current_tick_vals,
                ticktext=final_y_tick_text,
                tickfont=dict(family=font_family, size=tick_font_size, color="black"),
                ticks='outside',
                tickcolor='black',
                range=[y_pos - 1, y_pos + 2 if truth_features else y_pos + 1], 
                fixedrange=True
            ),
            plot_bgcolor='white',
            height=plot_height, 
            showlegend=True, 
            # Legend moved to the right-hand side
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02, # Just outside the plot area
                font=dict(size=legend_font_size)
            )
        )
        
        # Add shading for exact matches (Green)
        if exact_matches:
            for start, end in exact_matches:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=tp_color, opacity=tp_opacity,
                    layer="below", line_width=0,
                )
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=tp_color, opacity=tp_opacity, symbol='square'),
                name=match_label,
                hoverinfo='none'
            ))
        
        # Add shading for partial matches
        if partial_matches:
            for pos in partial_matches:
                # Only draw orange line if this position isn't part of an exact match
                if not (truth_set_flattened[pos] in exact_matches):
                    fig.add_vline(x=pos, line_width=partial_line_thickness, line_dash="solid", line_color=partial_color)
                    partial_drawn = True
            
            if partial_drawn:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color=partial_color, width=partial_line_thickness),
                    name="Partial match",
                    hoverinfo='none'
            ))
        
        # Add shading for false positives
        if no_matches:
            for start, end in no_matches:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=fp_color, opacity=fp_opacity, 
                    layer="below", line_width=0,
                )
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=fp_color, opacity=fp_opacity, symbol='square'),
                name=no_match_label,
                hoverinfo='none'
            ))
        
        # Add shading for margins
        if shade_margin > 0:
            fig.add_vrect(
                x0=seq_bounds[0], x1=seq_bounds[0] + shade_margin - 1,
                fillcolor=not_covered_color, opacity=not_covered_opacity,
                layer="below", line_width=0,
            )
            fig.add_vrect(
                x0=seq_bounds[1] - shade_margin, x1=seq_bounds[1],
                fillcolor=not_covered_color, opacity=not_covered_opacity,
                layer="below", line_width=0,
            )
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=not_covered_color, opacity=not_covered_opacity, symbol='square'),
                name=margin_label,
                hoverinfo='none'
            ))
        
        return fig


    def apply_copyable_seq_id_wrapper(html_content: str,
                                      seq_id_color: str = seq_ID_color,
                                      body_background_color: str = body_background_color,
                                      shadow_color: str = shadow_color, 
                                      shadow_intensity: str = shadow_intensity,
                                      plot_width_percent: str = plot_width_percent,
                                      note: str = None,
                                      note_font_size: int = int(12*font_scaler),
                                      font_scaler: float = font_scaler) -> str:
        """
        This function takes the generated Plotly HTML content and wraps it with a custom
        HTML structure that includes a container just below the plot with a copyable sequence id.
        The custom wrapper includes additional styling and functionality such as a copy button for the sequence ID.

        Args:
            html_content: The original HTML content generated by Plotly.
            plotly_html: The Plotly HTML content to be wrapped.
            shadow_color: The string that determines the color of the box-shadow around the new division.
            shadow_intensity: The string that determines the size of the box-shadow around the new division.

        Returns:
            str: The modified HTML content with the custom wrapper applied.
        """
        
        custom_wrapper = textwrap.dedent(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif;
                        margin: 0; 
                        padding: 10px; 
                        background: {body_background_color}; 
                        display: flex;
                        flex-direction: column;
                        min-height: 100vh;
                    }}
                    
                    .plotly-shell {{
                        width: {plot_width_percent};
                        margin: 0 auto;
                        background: #ffffff;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: {shadow_intensity} {shadow_color};
                        border: 1px solid #d1d5da;
                        box-sizing: border-box;
                    }}

                    .plotly-container {{
                        width: 100%;
                        margin: 0 auto;
                    }}
                    #copy-container {{
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        background: #ffffff;
                        padding: 12px 25px;
                        border-radius: 10px;
                        box-shadow: {shadow_intensity} {shadow_color};
                        margin: 15px auto 5px auto;
                        border: 1px solid #d1d5da;
                        width: {plot_width_percent};
                        box-sizing: border-box;
                    }}

                    .label-group {{
                        display: flex;
                        align-items: center;
                        gap: 15px;
                    }}

                    span.label-text {{
                        color: #444;
                        font-size: {int(13*font_scaler)}px;
                        text-transform: uppercase;
                        font-weight: 700;
                    }}

                    #id-display {{ 
                        font-family: SFMono-Regular, Consolas, monospace;
                        background: #f0f7ff;
                        color: {seq_id_color};
                        padding: 6px 14px;
                        border-radius: 6px;
                        font-weight: 600;
                        border: 1px solid #b2d4ff;
                        font-size: {int(16*font_scaler)}px;
                    }}

                    #copy-btn {{
                        width: 100px;
                        height: 40px;
                        padding: 10px 10px;
                        background: #719465;
                        color: white;
                        border: 1px solid rgba(27,31,35,.15);
                        border-radius: 6px;
                        cursor: pointer;
                        font-weight: 600;
                        transition: 0.2s;
                    }}

                    #copy-btn:hover {{ background: #618057; }}
                    #copy-btn.success {{ background: {seq_id_color}; }}
                </style>
            </head>
            <body>

            <div class="plotly-shell" style="text-align: right;">
                <div class="plotly-container">
                    {html_content}
                </div>
                <p style="font-size: {note_font_size}px; font-style: italic;">{'* ' + note if note else ''}</p>
            </div>

            <div id="copy-container">
                <div class="label-group">
                    <span class="label-text">Current Sequence ID:</span>
                    <span id="id-display">Detecting...</span>
                </div>
                <button id="copy-btn" onclick="copyId()">Copy ID</button>
            </div>

            <script>
            function copyId() {{
                const text = document.getElementById('id-display').innerText;
                const btn = document.getElementById('copy-btn');
                
                navigator.clipboard.writeText(text).then(() => {{
                    const originalText = btn.innerText;
                    btn.innerText = "✓ Copied";
                    btn.classList.add('success');
                    setTimeout(() => {{
                        btn.innerText = originalText;
                        btn.classList.remove('success');
                    }}, 1500);
                }});
            }}

            function initializeWatcher() {{
                const gd = document.querySelector('.plotly-graph-div');
                if (!gd) {{ 
                    setTimeout(initializeWatcher, 100); 
                    return; 
                }}

                const display = document.getElementById('id-display');

                const syncLabel = () => {{
                    try {{
                        const menu = gd.layout.updatemenus[0];
                        const activeIndex = (menu.active !== undefined && menu.active !== -1) ? menu.active : 0;
                        let label = menu.buttons[activeIndex].label;
                        
                        if (label) {{
                            const cleanedLabel = label.replace(/^\d+\)\s/, "");
                            display.innerText = cleanedLabel;
                        }}
                    }} catch (e) {{
                        console.log("Copyable sequence ID sync error.");
                    }}
                }};

                syncLabel();

                const events = ['plotly_restyle', 'plotly_relayout', 'plotly_update'];
                events.forEach(ev => {{
                    gd.on(ev, () => setTimeout(syncLabel, 100));
                }});
            }}

            window.addEventListener('load', initializeWatcher);
            </script>
            </body>
            </html>"""
        )
        
        return custom_wrapper


    def add_update_HTML_title(html_content: str, 
                            new_title: str) -> str:
        """
        Helper function to update the title field of the HTML report generated by Plotly after the fact.
        """
        
        # Find the start and end of the <head> tag
        head_start = html_content.find('<head>')
        head_end = html_content.find('</head>')
        
        if head_start == -1:
            # If there's no <head> tag, create one
            html_content = f"<head><title>{new_title}</title></head>{html_content}"
        else:
            # Extract the content within the <head> tag
            head_content = html_content[head_start + len('<head>'):head_end]
            
            # Find the start and end of the <title> tag within the <head> content
            title_start = head_content.find('<title>')
            title_end = head_content.find('</title>')
            
            if title_start != -1 and title_end != -1:
                # If there's an existing <title> tag, update its content
                title_tag = f"<title>{new_title}</title>"
                head_content = head_content[:title_start] + title_tag + head_content[title_end + len('</title>'):]
            else:
                # If there's no <title> tag, create a new one
                head_content = f"<title>{new_title}</title>" + head_content
                
            # Reconstruct the HTML content with the updated <head>
            html_content = html_content[:head_start + len('<head>')] + head_content + html_content[head_end:]
            
        return html_content


    def inject_base64_favicon(html_content: str, 
                              png_path: str) -> str:
        """
        Adds an embedded (base64 string) favicon to the head of the HTML report next to the tile
        and makes a head to put this in if none already exists.
        """
        
        # Read the PNG file and encode it to base64
        with open(png_path, "rb") as image_file:
            base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create the favicon link tag with the base64 encoded image
        favicon_link = f'<link rel="icon" type="image/png" href="data:image/png;base64,{base64_encoded_image}" />'
        
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the <head> tag and insert favicon string, else make one and insert
        head_tag = soup.find('head')
        if head_tag:
            head_tag.insert(0, BeautifulSoup(favicon_link, 'html.parser'))
        else:
            new_head_tag = soup.new_tag('head')
            new_head_tag.append(BeautifulSoup(favicon_link, 'html.parser'))
            soup.insert(0, new_head_tag)

        return str(soup)
    

    def modify_plotly_graph_div(html_content: str,
                                class_name: str = "plotly-graph-div",
                                height_added: int = 25) -> str:
        """
        Finds the Plotly plot <div> in the HTML report string and modifies the height parameter
        to add margin between x-axis label and plot note (if any). Modifying the source of this
        height parameter on the Plotly/Python end simply increase the plot height. This function
        only increases the container/div height without changing the Plotly plot height.
        """
        
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the specific div by its class
        div = soup.find('div', class_=class_name)
        
        if div:
            # Extract and split the style attribute
            style = div.get('style', '')
            style_parts = style.split(';')
            
            # Find the height part and adjust it
            for i, part in enumerate(style_parts):
                if 'height:' in part:
                    current_height = int(part.split(':')[1].replace('px', ''))
                    new_height = current_height + height_added
                    style_parts[i] = f'height:{new_height}px'
            
            # Join the style parts back together and update the style
            new_style = ';'.join(style_parts)
            div['style'] = new_style
        
        # Return the modified HTML content
        return str(soup)
    

    def cleanup_HTML(html_content: str,
                     indent_spaces: int = 4) -> str:
        """
        Wrapper that uses BeautifulSoup to improve whitespace/indent/readability of HTML content
        """
        
        raw = BeautifulSoup(html_content, 'html.parser')
        formatter = HTMLFormatter(indent=indent_spaces)
        improved = raw.prettify(formatter=formatter)
        return improved


    def update_HTML(html_content: str,
                    save_path: str,
                    function_calls_list: list,
                    verbose: bool = False):
        """
        Applies each of a list of (function, kwarg) tuples to an HTML content (generated by Plotly) and saves the changes.
        """
        
        # Apply each function/modifier to the file
        for function, kwargs in function_calls_list:
            html_content = function(html_content, **kwargs)
        
        # Write the modified HTML content back to the file
        try:
            with open(save_path, 'w', encoding='utf-8') as file:
                file.write(cleanup_HTML(html_content))
            if verbose:
                print(Fore.GREEN + f"Custom styles and modifications added successfully to '{save_path}'.")
        except Exception as e:
            print(Fore.RED + f"An error occurred while writing to the file: {e}")
            exit()



    ### MAIN PROCEDURE: VISUAL REPORT GENERATION
    
    print(Fore.MAGENTA + "Preparing Plotly HTML report...")
    
    # Initialize the figure and loop variables
    msg_length = 0
    final_fig = go.Figure()
    dropdown_buttons = []
    trace_index_counter = 0

    # List and sort the prediction GFF files, so the final dropdown is ordered
    gff_files = sorted([f.name for f in predictions_dir.glob('*.gff')])
    
    # Check the provided reference annotation GFF for a singular value in the source field to use in the plots
    if truth_features_path and not truth_source_name:
        command = f"cat {truth_features_path} | cut -f 2 | sort | uniq"
        truth_source = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.split('\n')
        truth_source = [source for source in truth_source if source]
        if len(truth_source) != 1:
            print(Fore.YELLOW + f"WARNING: No unanimous 'source' name found for features in '{truth_features_path}', will proceed with placeholder 'Reference'.")
            truth_source = 'Reference'
        else:
            truth_source = truth_source[0]
    elif truth_source_name:
        truth_source = truth_source_name
    
    ## Iterate through all prediction GFFs and visualize results (against ground truth if available)
    first_plot_flag = True
    skip_counter = 0
    for i, prediction in enumerate(gff_files):
        
        # Pull the prefix/seq_id from the prediction file name and import/parse HEX-finder predictions
        file_id = prediction.replace('.gff', '')
        seq_id, seq_source, seq_type, seq_length, predicted_exons = parse_gff(predictions_dir / prediction)
            
        # This is a sanity check that should not be need unless the tool was being altered or misused.
        if seq_id != file_id:
            print(Fore.RED + f"ERROR: Prefix of file '{prediction}' does not match {seq_id} in its 'seqid' field (first column). This will cause issues. Stopping.")
            exit()
        
        # Let the user know the plot is being created for the current file.
        if not quiet:
            print(Fore.MAGENTA + f"Processing profile {i+1} of {len(gff_files)} ('{prediction}')...")
        
        # If a path to a GFF with local truth features has been provided, then import the relevant ones for this sequence from the GFF
        if truth_features_path:
            
            # Separate out the relevant reference/truth annotation for this sequence
            command = f'grep "{seq_id}" {truth_features_path} > {temp_dir / "reference_features_temp.gff"}'
            fetch_strand_annotation = subprocess.run(command, shell=True)
            if not quiet:
                check_command_exit(fetch_strand_annotation,
                                Fore.YELLOW + f"WARNING: {seq_id} not found in '{truth_features_path}'.\nThere may truly be no reference features for this sequence, or the sequence IDs used in the reference file are not compatibly formatted.",
                                stop_after_msg=False)

            # Import the features (if any) from the reference GFF file for this sequence
            _, _, _, _, truth_features = parse_gff(temp_dir / "reference_features_temp.gff")
            truth_dict = {'source': truth_source,
                        'features': [ (feature[0], feature[1], feature[3][truth_labels_field] if truth_labels_field and truth_labels_field in feature[3] else None) for feature in truth_features ],
                        'color': 'black'} 
        else:
            truth_dict=None
        
        # If the user does not want to show empty plots (no reference exons plus no predictions), then skip making this plot
        if skip_empty and (not truth_dict or not truth_features) and (not predicted_exons):
            if not quiet:
                print(Fore.YELLOW + f'Skipped as per user request (-se/--skip_empty): no predicted or reference features to plot.')
            skip_counter += 1
            continue
        
        else:
            # Generate the figure but DO NOT show it yet
            get_coords = lambda exons_list: [exon_tuple[:2] for exon_tuple in exons_list] if exons_list else []
            temp_fig = create_feature_lane_plot(get_coords(predicted_exons), 
                                                seq_id=seq_id,
                                                seq_length=seq_length, 
                                                truth_features=truth_dict,
                                                add_note=True,
                                                **plotly_colors)
            
            
            ## UPDATE THE MAIN FIGURE WITH TRACES FROM THIS SEQUENCE
            
            # Capture the layout style (axis lines, fonts, etc.) from the FIRST plot
            # This ensures the final report looks exactly like the individual plots
            if first_plot_flag:
                final_fig.update_layout(temp_fig.layout)
                first_plot_flag = False

            # Add all traces from the temp_fig to the final_fig
            # Only set visible=True for the FIRST sequence (i==0)
            num_traces = len(temp_fig.data)
            for trace in temp_fig.data:
                trace.visible = (i == 0)
                final_fig.add_trace(trace)

            # Capture the shapes (highlights) and layout specifics
            # temp_fig.layout.shapes contains the green/red rectangles and orange lines
            current_shapes = temp_fig.layout.shapes
            current_title_text = temp_fig.layout.title.text 
            
            # Explicitly define the correct x range based on sequence length
            current_xaxis_range = [1, seq_length]
            
            # Create a dropdown button for this sequence
            button = dict(label=f'{i+1}) {file_id}',
                            method="update",
                            args=[
                                    {"visible": []}, # Placeholder for later
                                    {
                                        "title.text": current_title_text,
                                        "shapes": current_shapes,
                                        "xaxis.range": current_xaxis_range, 
                                        "xaxis.minallowed": current_xaxis_range[0],
                                        "xaxis.maxallowed": current_xaxis_range[1],
                                        "xaxis.autorange": True
                                    }
                                ],
                                # Helper key to store which indices belong to this file
                                _indices=range(trace_index_counter, trace_index_counter + num_traces)
                        )
            dropdown_buttons.append(button)
            
            # Increment counter so next file knows where its traces start
            trace_index_counter += num_traces
    
    
    ## FINALIZE THE REPORT AND SAVE TO HTML
    
    if dropdown_buttons:
        # Fix the "visible" mask for each button
        # Now that we know the TOTAL number of traces, we can build the True/False lists
        total_traces = trace_index_counter
        
        for button in dropdown_buttons:
            visibility_mask = [False] * total_traces
            for idx in button["_indices"]:
                visibility_mask[idx] = True
            
            button["args"][0]["visible"] = visibility_mask
            del button["_indices"] # Clean up helper key

        # Add the drop down menu to the final figure
        final_fig.update_layout(updatemenus=[dict(
                                            active=0,
                                            buttons=dropdown_buttons,
                                            direction="down",
                                            pad={"r": 10, "t": 10},
                                            showactive=True,
                                            x=0.0, # Left aligned with the axis
                                            xanchor="left",
                                            y=1.35, # Positioned above plot but below title
                                            yanchor="top",
                                            font=dict(size=dropdown_font_size, weight='bold', color=seq_ID_color)
                                        )
                                    ],
                                    # Add Top Margin (t) to make space for Title + Dropdown
                                    margin=dict(t=150),
                                    # Ensure the FIRST view also has the correct range and autorange disabled
                                    xaxis=dict(
                                        range=dropdown_buttons[0]["args"][1]["xaxis.range"],
                                        autorange=False
                                    )
                                )
        
        # Set initial visibility for the first button
        initial_visibility = dropdown_buttons[0]["args"][0]["visible"]
        for trace, visible in zip(final_fig.data, initial_visibility):
            trace.visible = visible
        
        # Removes buttons that do not work in the context of this HTML report/dropdown and makes remaining ones visible by default
        # This prevents users from accidentally reverting to the view inherited from the first trace/plot for an incompatible set of features.
        plot_config = {'displaylogo': False, 
                       'modeBarButtonsToRemove': ['resetScale2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d'], # 'pan2d'
                       'displayModeBar': True,
                       'doubleClick': 'autosize',
                       'showTips' : True}
        
        # Write the HTML report and make some style and function modifications to the page after-the-fact
        # This is a little bit hacky, but some non-Plotly-controlled page formatting changes can be easily applied this way
        if not quiet:
            print(Fore.MAGENTA + "Saving HTML report...")
        plotly_html = pio.to_html(final_fig, config=plot_config, full_html=False, include_plotlyjs=include_javascript)
        update_HTML(html_content = plotly_html,
                    save_path = output_path, 
                    function_calls_list = [ (apply_copyable_seq_id_wrapper, {'note' : 'Attempting to pan past the sequence boundaries forces a zoom instead, which looks glitchy but updates correctly after mouse release. Most exons require zooming to be visible for sequences approaching ≥ 500 Knc in length.'}),
                                            (add_update_HTML_title, {'new_title' : "HEX-finder: Predictions Report"}),
                                            (modify_plotly_graph_div, {'height_added' : 25}),
                                            (inject_base64_favicon, {'png_path' : favicon_path})])
        
        # Notify user how many sequences were skipped due to no features (truth or predicted) for display
        if skip_counter != 0:
            print(Fore.YELLOW + f"NOTE: {skip_counter} sequences were omitted from the report due to no HEX-finder predictions and no reference features (-se/--skip_empty).")
            print(Fore.YELLOW + f"--> You can rerun with both -se and -v to see or log which ones were skipped.")
        print(Fore.GREEN + f"Done! Report was written to '{output_path}'.")
    
    else:
        print(Fore.YELLOW + "WARNING: No valid data found to generate a visual report.")
    
    # Clear the temporary folder
    if os.listdir(temp_dir):
        shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
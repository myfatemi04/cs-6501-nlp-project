# From https://github.com/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb

# This is some CSS (tells us what style )to give each token a thin gray border, to make it easy to see token separation
style_string = """<style> 
    span.token {
        border: 1px solid rgb(123, 123, 123);
        color: black;
        } 
    </style>"""
style_string = ''

def calculate_color(val, max_val, min_val):
    # Hacky code that takes in a value val in range [min_val, max_val], normalizes it to [0, 1] and returns a color which interpolates between slightly off-white and red (0 = white, 1 = red)
    # We return a string of the form "rgb(240, 240, 240)" which is a color CSS knows
    normalized_val = (val - min_val) / (max_val - min_val)
    return f"rgb(240, {240*(1-normalized_val)}, {240*(1-normalized_val)})"

def calculate_signed_color(val, scale):
    # Hacky code that takes in a value val in range [min_val, max_val], normalizes it to [0, 1] and returns a color which interpolates between slightly off-white and red (0 = white, 1 = red)
    # We return a string of the form "rgb(240, 240, 240)" which is a color CSS knows
    normalized_val = val / scale

    # if negative, plot as red
    if normalized_val < 0:
        normalized_val = -normalized_val
        normalized_val = min(normalized_val, 1)
        return f"rgb(240, {240*(1-normalized_val)}, {240*(1-normalized_val)})"
    
    # if positive, plot as green
    normalized_val = min(normalized_val, 1)
    return f"rgb({240*(1-normalized_val)}, 240, {240*(1-normalized_val)})"

def basic_neuron_vis(tokens, acts, layer, neuron_index, max_val=None, min_val=None):
    """
    text: The text to visualize
    layer: The layer index
    neuron_index: The neuron index
    max_val: The top end of our activation range, defaults to the maximum activation
    min_val: The top end of our activation range, defaults to the minimum activation

    Returns a string of HTML that displays the text with each token colored according to its activation

    Note: It's useful to be able to input a fixed max_val and min_val, because otherwise the colors will change as you edit the text, which is annoying.
    """
    act_max = acts.max()
    act_min = acts.min()
    # Defaults to the max and min of the activations
    if max_val is None:
        max_val = act_max
    if min_val is None:
        min_val = act_min
    # We want to make a list of HTML strings to concatenate into our final HTML string
    # We first add the style to make each token element have a nice border
    htmls = [style_string]
    # We then add some text to tell us what layer and neuron we're looking at - we're just dealing with strings and can use f-strings as normal
    # h4 means "small heading"
    htmls.append(f"<h4>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron_index}</b></h4>")
    # We then add a line telling us the limits of our range
    htmls.append(
        f"<h4>Range: <b>[{act_min:.4f}, {act_max:.4f}]</b></h4>"
    )
    # If we added a custom range, print a line telling us the range of our activations too.
    if act_max != max_val or act_min != min_val:
        htmls.append(
            f"<h4>Custom Range Set: <b>[{min_val:.4f}, {max_val:.4f}]</b></h4>"
        )
    htmls.append("<pre>")
    # Convert the text to a list of tokens
    str_tokens = tokens
    for tok, act in zip(str_tokens, acts):
        # A span is an HTML element that lets us style a part of a string (and remains on the same line by default)
        # We set the background color of the span to be the color we calculated from the activation
        # We set the contents of the span to be the token
        tok = tok.replace(" ", "&nbsp;")
        htmls.append(
            f"<span class='token' style='background-color:{calculate_color(act, min_val, max_val)}'>{tok}</span>"
        )
    htmls.append("</pre>")

    return "".join(htmls)


def basic_neuron_vis_signed(tokens, acts, scale):
    """
    text: The text to visualize
    layer: The layer index
    neuron_index: The neuron index
    max_val: The top end of our activation range, defaults to the maximum activation
    min_val: The top end of our activation range, defaults to the minimum activation

    Returns a string of HTML that displays the text with each token colored according to its activation

    Note: It's useful to be able to input a fixed max_val and min_val, because otherwise the colors will change as you edit the text, which is annoying.
    """
    # We want to make a list of HTML strings to concatenate into our final HTML string
    # We first add the style to make each token element have a nice border
    htmls = [style_string]
    # We then add some text to tell us what layer and neuron we're looking at - we're just dealing with strings and can use f-strings as normal
    # h4 means "small heading"
    # htmls.append(f"<h4>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron_index}</b></h4>")
    # We then add a line telling us the limits of our range
    # htmls.append(
    #     f"<h4>Range: <b>[{act_min:.4f}, {act_max:.4f}]</b></h4>"
    # )
    # If we added a custom range, print a line telling us the range of our activations too.
    # if act_max != max_val or act_min != min_val:
    #     htmls.append(
    #         f"<h4>Custom Range Set: <b>[{min_val:.4f}, {max_val:.4f}]</b></h4>"
    #     )
    htmls.append("<pre style='color: black'>")
    # Convert the text to a list of tokens
    str_tokens = tokens
    for tok, act in zip(str_tokens, acts):
        # A span is an HTML element that lets us style a part of a string (and remains on the same line by default)
        # We set the background color of the span to be the color we calculated from the activation
        # We set the contents of the span to be the token
        tok = tok.replace(" ", "&nbsp;")
        htmls.append(
            f"<span class='token' style='background-color:{calculate_signed_color(act, scale)}'>{tok}</span>"
        )
    htmls.append("</pre>")

    return "".join(htmls)

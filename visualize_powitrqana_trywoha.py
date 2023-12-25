import pandas as pd
from io import StringIO
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from io import StringIO


data = """
     Hour  DayOfWeek  Probability_True
86      0          1          0.042921
93      1          1          0.010534
100     2          1          0.000000
107     3          1          0.003478
114     4          1          0.161665
121     5          1          0.050571
128     6          1          0.000000
135     7          1          0.000000
142     8          1          0.000000
149     9          1          0.000000
156    10          1          0.025468
163    11          1          0.054379
3      12          1          0.077497
10     13          1          0.079167
17     14          1          0.025574
24     15          1          0.086104
31     16          1          0.021818
38     17          1          0.000000
45     18          1          0.000000
52     19          1          0.000000
59     20          1          0.000000
66     21          1          0.000000
73     22          1          0.000000
80     23          1          0.000000
87      0          2          0.000000
94      1          2          0.000000
101     2          2          0.000000
108     3          2          0.000000
115     4          2          0.000000
122     5          2          0.000000
129     6          2          0.000000
136     7          2          0.000000
143     8          2          0.000000
150     9          2          0.000000
157    10          2          0.000000
164    11          2          0.000000
4      12          2          0.000000
11     13          2          0.015901
18     14          2          0.074138
25     15          2          0.057497
32     16          2          0.108609
39     17          2          0.046806
46     18          2          0.000000
53     19          2          0.019700
60     20          2          0.000000
67     21          2          0.036029
74     22          2          0.024762
81     23          2          0.081316
88      0          3          0.040071
95      1          3          0.000000
102     2          3          0.069839
109     3          3          0.154505
116     4          3          0.076172
123     5          3          0.099552
130     6          3          0.032385
137     7          3          0.000000
144     8          3          0.000000
151     9          3          0.000000
158    10          3          0.000000
165    11          3          0.000000
5      12          3          0.073176
12     13          3          0.116233
19     14          3          0.084893
26     15          3          0.023452
33     16          3          0.000000
40     17          3          0.000000
47     18          3          0.000000
54     19          3          0.000000
61     20          3          0.034337
68     21          3          0.003533
75     22          3          0.007584
82     23          3          0.000000
89      0          4          0.000000
96      1          4          0.000000
103     2          4          0.000000
110     3          4          0.000000
117     4          4          0.031208
124     5          4          0.058068
131     6          4          0.035392
138     7          4          0.045603
145     8          4          0.030074
152     9          4          0.004619
159    10          4          0.000000
166    11          4          0.051794
6      12          4          0.062099
13     13          4          0.039376
20     14          4          0.055255
27     15          4          0.025614
34     16          4          0.010446
41     17          4          0.032460
48     18          4          0.037395
55     19          4          0.000000
62     20          4          0.057554
69     21          4          0.000000
76     22          4          0.000000
83     23          4          0.050108
90      0          5          0.061561
97      1          5          0.003658
104     2          5          0.000000
111     3          5          0.000000
118     4          5          0.028437
125     5          5          0.024795
132     6          5          0.012313
139     7          5          0.097213
146     8          5          0.037333
153     9          5          0.014295
160    10          5          0.031946
167    11          5          0.119370
0      12          5          0.080831
7      13          5          0.069292
14     14          5          0.048866
21     15          5          0.000000
28     16          5          0.000000
35     17          5          0.030313
42     18          5          0.059359
49     19          5          0.038852
56     20          5          0.010494
63     21          5          0.009063
70     22          5          0.054421
77     23          5          0.088304
84      0          6          0.120918
91      1          6          0.071618
98      2          6          0.075701
105     3          6          0.101176
112     4          6          0.105519
119     5          6          0.059823
126     6          6          0.046679
133     7          6          0.053805
140     8          6          0.098575
147     9          6          0.010681
154    10          6          0.006951
161    11          6          0.091061
1      12          6          0.172998
8      13          6          0.151546
15     14          6          0.127667
22     15          6          0.049736
29     16          6          0.000000
36     17          6          0.000000
43     18          6          0.000000
50     19          6          0.000000
57     20          6          0.014421
64     21          6          0.000000
71     22          6          0.012876
78     23          6          0.050038
85      0          7          0.107891
92      1          7          0.148430
99      2          7          0.112093
106     3          7          0.053081
113     4          7          0.072262
120     5          7          0.113387
127     6          7          0.062286
134     7          7          0.000000
141     8          7          0.000000
148     9          7          0.000000
155    10          7          0.043273
162    11          7          0.022055
2      12          7          0.000000
9      13          7          0.055901
16     14          7          0.042161
23     15          7          0.000000
30     16          7          0.000000
37     17          7          0.000000
44     18          7          0.000000
51     19          7          0.000000
58     20          7          0.000000
65     21          7          0.000000
72     22          7          0.041173
79     23          7          0.001032
"""

# Using StringIO to create a file-like object for pandas read_csv
df = pd.read_csv(StringIO(data), delim_whitespace=True)

df_sorted = df.sort_values(by='Probability_True', ascending=False)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df_sorted)
# Reset options to default after printing
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')


# Create the main application window
app = tk.Tk()
app.title("Weather Forecast")

# Frame for the weather forecast
frame = ttk.Frame(app, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Matplotlib figure
fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
ax.plot(df['Hour'], df['Probability_True'], marker='o', linestyle='-', color='b')
ax.set_xlabel('Hour')
ax.set_ylabel('Probability of Rain')
ax.set_title('Weather Forecast')

# Canvas for embedding Matplotlib figure in Tkinter
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Run the Tkinter event loop
app.mainloop()
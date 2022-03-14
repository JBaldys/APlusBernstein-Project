# Alliance Bernstein Factor Timing Project
Our team’s overarching goal of this project would be to identify a unique, comprehensive, and actionable trading strategy that will meet the following standards: 1.) outperforms a passive S&P 500 strategy, 2.) incorporates a variety of different underlying signals/financial data, most notably sentiment and volatility-based signals, 3.) presents a tangible competitive value, and 4.) represents the nature of future research that we believe Alliance Bernstein can conduct beyond the scope of this individual project.

### Background Information (AB provided materials)
https://vanderbilt365-my.sharepoint.com/:b:/r/personal/peng_zhang_vanderbilt_edu/Documents/Teams22/Client%20Projects/AB/FactorTimingABVU2022.pdf?csf=1&web=1&e=N8B2AD

https://vanderbilt365-my.sharepoint.com/:f:/g/personal/jessica_l_baldys_vanderbilt_edu/Evsfd7A4DvNGu4kwD8FD32cB6Ht4LOMENqrJz1Y-QpUSsw?e=LOj2mc

### External Resources
The Investor's Podcast- Factor Investing - https://alphaarchitect.com/2019/11/22/the-investors-podcast-factor-investing-jack/

Investopedia - Factor Investing - https://www.investopedia.com/terms/f/factor-investing.asp

Factor Timing- Keep it Simple - https://www.researchaffiliates.com/publications/articles/828-factor-timing-keep-it-simple#:~:text=Factor%20timing%20is%20the%20ability,a%20factor's%20discount%20and%20momentum.

Quantopian Lecture Series - Fundamental Factor Models - https://www.youtube.com/watch?v=P16zDtf0CE0&list=PLRFLF1OxMm_UL7WUWM31iynp0jMVf_vLW&index=11

Quantopian Lecture Series - Factor Analysis - https://www.youtube.com/watch?v=v5IYcBxMDYE&list=PLRFLF1OxMm_UL7WUWM31iynp0jMVf_vLW&index=18

(see other videos from Quantopian too!)

# Tentative Timeline / Important Dates
-	3/14-3/19: EDA, feature engineering
-	3/18: In-person visit with Che from Alliance Bernstein, 2-3 pm (prioritize making this meeting!)
-	3/20-3/26: EDA cont., begin modeling
-	3/27-4/2: modeling, beginning of model revision / novel signal creation (all group members should have already been doing research into novel signal creation up until this point)
-	4/3-4/16: “flex time”, issues during this time will be dependent on what group members feel their bandwidth looks like at this point
-	4/17-4/30: presentation creation and paper writing
-	5/2: presentation practice, final revisions, final presentation

# Database Login Info

- Bloomberg
-   Username:
-   Password:
- Factset
-   Username:
-   Password:
- CapIQ
-   Username:
-   Password:
- Refinitiv (can access Refinitiv data through the Eikon API)
-   Username:
-   Password:
- CRSP
-   Username:
-   Password:

# Zoom Link
Jessica Baldys is inviting you to a scheduled Zoom meeting.

Topic: Alliance Bernstein
Time: This is a recurring meeting Meet anytime

Join Zoom Meeting
https://vanderbilt.zoom.us/j/5641443704

Meeting ID: 564 144 3704
One tap mobile
+16465588656,,5641443704# US (New York)
+13462487799,,5641443704# US (Houston)

Dial by your location
        +1 646 558 8656 US (New York)
        +1 346 248 7799 US (Houston)
Meeting ID: 564 144 3704
Find your local number: https://vanderbilt.zoom.us/u/aFD27W2kD

Join by SIP
5641443704@zoomcrc.com

Join by H.323
162.255.37.11 (US West)
162.255.36.11 (US East)
115.114.131.7 (India Mumbai)
115.114.115.7 (India Hyderabad)
213.19.144.110 (Amsterdam Netherlands)
213.244.140.110 (Germany)
103.122.166.55 (Australia Sydney)
103.122.167.55 (Australia Melbourne)
149.137.40.110 (Singapore)
64.211.144.160 (Brazil)
149.137.68.253 (Mexico)
69.174.57.160 (Canada Toronto)
65.39.152.160 (Canada Vancouver)
207.226.132.110 (Japan Tokyo)
149.137.24.110 (Japan Osaka)
Meeting ID: 564 144 3704

# Contact Info

Andrew Chin Email: andrew.chin@alliancebernstein.com

Che Guan Email: che.guan@alliancebernstein.com

Anubha Nagar Email: anubha.nagar@vanderbilt.edu Phone: 615-674-3177

Jessica Baldys Email: jessica.l.baldys@vanderbilt.edu Phone: 708-603-6558

KC Barrett Email: kevin.c.barrett@vanderbilt.edu Phone: 401-533-2480

Qiushi Yan Email: qiushi.yan@vanderbilt.edu Phone: 629-239-9151

## Packages

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* great expectations
* h2o
* fastai
* huggingface
* datasets
* quantopian - https://github.com/quantopian

# Meetings

- Will meet 2x a week - Thursday @4:30pm and Sunday @10:30am
- Thursday meetings serve as mid-point check-ins for issues
- All pull requests / code reviews should be entered by the Saturday prior to Sunday meetings
- All issues should be closed on the Sunday they are due


# Repo Structure 

The repo is structured as follows: Notebooks are grouped according to their series (e.g., 10, 20, 30, etc) which reflects the general task to be performed in those notebooks.  Start with the *0 notebook in the series and add other investigations relevant to the task in the series (e.g., `11-cleaned-scraped.ipynb`).  If your notebook is extremely long, make sure you've utilized nbdev reuse capabilities and consider whether you can divide the notebook into two notebooks.

All files which appear in the repo should be able to run, and not contain error or blank cell lines, even if they are relatively midway in development of the proposed task. All notebooks relating to the analysis should have a numerical prefix (e.g., 31-) followed by the exploration (e.g. 31-text-labeling). Any utility notebooks should not be numbered, but be named according to their purpose. All notebooks should have lowercase and hyphenated titles (e.g., 10-process-data not 10-Process-Data). All notebooks should adhere to literate programming practices (i.e., markdown writing to describe problems, assumptions, conclusions) and provide adequate although not superfluous code comments.

# Extra Tutorials
* **Python usage**: Whirlwind Tour of Python, Jake VanderPlas ([Book](https://learning.oreilly.com/library/view/a-whirlwind-tour/9781492037859/), [Notebooks](https://github.com/jakevdp/WhirlwindTourOfPython))
* **Data science packages in Python**: [Python Data Science Handbook, Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/) 
* **HuggingFace**: [Website](https://huggingface.co/transformers/index.html), [Course/Training](https://huggingface.co/course/chapter1), [Inference using pipelines](https://huggingface.co/transformers/task_summary.html), [Fine tuning models](https://huggingface.co/transformers/training.html)
* **fast.ai**: [Course](https://course.fast.ai/), [Quick start](https://docs.fast.ai/quick_start.html)
* **h2o**: [Resources, documentation, and API links](https://docs.h2o.ai/#h2o)
* **nbdev**: [Overview](https://nbdev.fast.ai/), [Tutorial](https://nbdev.fast.ai/tutorial.html)
* **Git tutorials**: [Simple Guide](https://rogerdudler.github.io/git-guide/), [Learn Git Branching](https://learngitbranching.js.org/?locale=en_US)
* **ACCRE how-to guides**: [DSI How-tos](https://github.com/vanderbilt-data-science/how-tos)  

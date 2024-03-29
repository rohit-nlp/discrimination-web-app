# Discrimination Web App - Detecting discrimination through Suppes-Bayes Causal Network
**A bacherlor thesis by Blai Ras**. Read the full paper at https://drive.google.com/file/d/12ta5mAYn_H_jlzm7jM6DsEvHPYLxJsX6/view?usp=sharing

## Running the Project Locally

### Essential requirements

First, install git:

```bash
	sudo apt install git
```

Second, check your current Python version with:

```bash
	python3 -V
```

**This project needs Python >= 3.5**

Third, install pip3:

```bash
	sudo apt install python3-pip
```

### Project specific requirments

Install R on your computer. Check if your already have it:

```bash
	R --version
```

**This project needs R >= 3.2.3**


If not, first add the project GNU Privacy Guard key 

```bash
	sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
```

Add the R Repository:

```bash
	sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
```

Update & install:

```bash
	sudo apt update
	sudo apt install r-base
```

*On the first run project will tell you if you want to install the required bnlearn R library. You can type 'yes' or install it manually on R before launching the project, with:*

```bash
		install.packages(“bnlearn”)
```

---

Clone the repository to your local machine:

```bash
	git clone https://github.com/rohit-nlp/discrimination-web-app.git
```

---

Install the requirements:

```bash
	pip3 install -r requirements.txt
```

*Some of the requirements cannot be installed through pip.*

**Pycairo**

```bash
	pip3 install pycairo
	pip3 install cairocffi
```
or

```bash
	sudo apt-get install python3-cairocffi
```

**iGraph**

1.
```bash
	pip3 install python-igraph
```

If (1) doesn't work try:

```bash
	sudo apt install libxml2-dev libz-dev
```

And try (1) again.

---

Apply the migrations:

```bash
	python3 manage.py migrate
```

Finally, run the development server:

```bash
	python3 manage.py runserver
```

The project will be available at **http://127.0.0.1:8000/**

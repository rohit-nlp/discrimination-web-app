# Discrimination Web App - Detecting discrimination through Suppes-Bayes Causal Network, a bacherlor thesis by Blai Ras

## Running the Project Locally

###Essential requirements

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

###Project specific requirments

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


Clone the repository to your local machine:

```bash
	git clone https://github.com/rohit-nlp/discrimination-web-app.git
```

Install the requirements:

```bash
	pip install -r requirements.txt
```

Apply the migrations:

```bash
	python3 manage.py migrate
```

Finally, run the development server:

```bash
	python3 manage.py runserver
```

The project will be available at ****.


## License


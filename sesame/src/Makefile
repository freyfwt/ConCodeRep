.PHONY: all pull simopt crossopt clean distclean

###################
# GENERAL TARGETS #
###################

all: repos/checked_out \
     docgen/build/libs/docgen-all.jar \
     simtok/build/libs/simtok-all.jar \
     sample/build/libs/sample-all.jar \
     docs.db \
     docsim_calc \
     toksim_calc \
     sampled.csv \
     html_out

pull: repos/checked_out

simopt: repos/checked_out_simopt docgen/build/libs/docgen-all.jar bin/activate
	rm -f docs.db
	java -cp docgen/build/libs/docgen-all.jar -Dfilter=false -Dthreshold=8 DocGen repos/checkstyle \
	    repos/commons-collections repos/commons-lang repos/commons-math repos/deeplearning4j \
	    repos/eclipse.jdt.core repos/eclipse.swt repos/freemind repos/guava \repos/hsqldb \
	    repos/openjdk8 repos/trove
	mv docs.db docs-train.db
	source bin/activate && python3 simopt.py simopt.in

crossopt:
	[ -e docs-train.db ] || ( echo "Run make simopt first" && exit 1 )
	source bin/activate && python3 crossopt.py

clean:
	cd docgen && ./gradlew clean
	cd simtok && ./gradlew clean
	cd sample && ./gradlew clean
	cd methodextracter && ./gradlew clean

distclean: clean
	rm -Rf repos
	rm -f docs.db
	rm -f docs-train.db
	rm -f docsim_calc
	rm -f toksim_calc
	rm -f sampled.csv
	rm -Rf html_out
	rm -f raters
	rm -f assignments.csv
	rm -f pip-selfcheck.json
	rm -Rf bin include
	rm -Rf lib/python3*
	rm -Rf __pycache__
	[ ! -e lib64 ] || unlink lib64


###################
# SPECIAL TARGETS #
###################

repos:
	mkdir repos

repos/caffeine: | repos
	git clone "https://github.com/ben-manes/caffeine.git" $@

repos/checkstyle: | repos
	git clone "https://github.com/checkstyle/checkstyle.git" $@

repos/commons-collections: | repos
	git clone "http://git-wip-us.apache.org/repos/asf/commons-collections.git" $@

repos/commons-lang: | repos
	git clone "http://git-wip-us.apache.org/repos/asf/commons-lang.git" $@

repos/commons-math: | repos
	git clone "http://git-wip-us.apache.org/repos/asf/commons-math.git" $@

repos/deeplearning4j: | repos
	git clone "https://github.com/deeplearning4j/deeplearning4j.git" $@

repos/eclipse.jdt.core: | repos
	git clone "https://git.eclipse.org/r/jdt/eclipse.jdt.core" $@

repos/eclipse.swt: | repos
	git clone "https://git.eclipse.org/r/platform/eclipse.platform.swt" $@

repos/freemind: | repos
	git clone "git://git.code.sf.net/p/freemind/code" $@

repos/guava: | repos
	git clone "https://github.com/google/guava.git" $@

repos/hsqldb: | repos
	svn checkout -q https://svn.code.sf.net/p/hsqldb/svn/base/trunk $@

repos/openjdk8: | repos
	hg clone "http://hg.openjdk.java.net/jdk8/jdk8/jdk/" $@

repos/openjdk11: | repos
	hg clone "http://hg.openjdk.java.net/jdk/jdk11/" $@

repos/trove: | repos
	git clone "https://bitbucket.org/trove4j/trove.git" $@

repos/checked_out: repos/caffeine repos/checkstyle repos/commons-collections \
                   repos/commons-lang repos/commons-math repos/deeplearning4j \
                   repos/eclipse.jdt.core repos/freemind repos/guava repos/openjdk11 \
                   repos/trove | repos 
	git -C repos/caffeine checkout -q "f107fbc8b93883ec5513916c6c7fb5a819e6b498"
	git -C repos/checkstyle checkout -q "b2f8bca798cedd06065fc137d2bd5e5976e73cee"
	git -C repos/commons-collections checkout -q "74ad2114700446837ed451f36107a056301a65fc"
	git -C repos/commons-lang checkout -q "f110da945a8af0914f348cfe576b5a8257159fd9"
	git -C repos/commons-math checkout -q "34bd1707749c71302207fb2fc0559b6ca301d895"
	git -C repos/deeplearning4j checkout -q "19f269cab04c77cc33dcae7344b73a9afaa13dd6"
	git -C repos/eclipse.jdt.core checkout -q "ea5aa5d8a1f53c0e3f91e2a88f7287769c31a8b7"
	git -C repos/freemind checkout -q "39f6673b6a9a3da9ded46a2881d876dc84297634"
	git -C repos/guava checkout -q "581ba1436ebaa54a7f5d0f1db8cc4da0ca72127e"
	hg --cwd repos/openjdk11 update -r "1ddf9a99e4ad"
	git -C repos/trove checkout -q "3237c62b59245913a554ce1ed27c768aad7ac6ac"
	@touch repos/checked_out

repos/checked_out_simopt: repos/checkstyle repos/commons-collections repos/commons-lang \
                          repos/commons-math repos/deeplearning4j repos/eclipse.jdt.core \
                          repos/eclipse.swt repos/freemind repos/guava repos/hsqldb repos/openjdk8 \
                          repos/trove
	git -C repos/checkstyle checkout -q "a3e4bcc8b78326ee2fab31803ba0336461e10b6d"
	git -C repos/commons-collections checkout -q "edf3867212b2075696ee2dcf7cc8c7c4733d4d31"
	git -C repos/commons-lang checkout -q "78134f6b3f1facd019e604d2cd000c4ce7cf9a0a"
	git -C repos/commons-math checkout -q "b645f5d085f121ae82e480491a58a8d7753a34d0"
	git -C repos/deeplearning4j checkout -q "172836de6179a355774861f4052ca9bcd751d228"
	git -C repos/eclipse.jdt.core checkout -q "73770094c4f65d4f7343d820da475a1cbfa0ec9d"
	git -C repos/eclipse.swt checkout -q "3f312b5478026ef014a71590f6d63d909780f330"
	git -C repos/freemind checkout -q "1_0_1"
	git -C repos/guava checkout -q "0915b414d10d0737ac8fbd4819462d8141a09c8d"
	cd repos/hsqldb && svn update -q -r "r5656" && cd ../..
	hg --cwd repos/openjdk8 update -r "687fd7c7986d"
	git -C repos/trove checkout -q "a6ac7ec39440d61b66afcfe7d24ff4a9570bb621"
	@touch repos/checked_out_simopt


docgen/build/libs/docgen-all.jar:
	cd docgen && ./gradlew fatJar

simtok/build/libs/simtok-all.jar:
	cd simtok && ./gradlew fatJar

sample/build/libs/sample-all.jar:
	cd sample && ./gradlew fatJar

methodextracter/build/libs/methodextracter-all.jar:
	cd methodextracter && ./gradlew fatJar


docs.db: repos/checked_out docgen/build/libs/docgen-all.jar
	rm -f docs.db
	java -cp docgen/build/libs/docgen-all.jar DocGen repos/caffeine repos/checkstyle \
	     repos/commons-collections repos/commons-lang repos/commons-math repos/deeplearning4j \
	     repos/eclipse.jdt.core repos/freemind repos/guava repos/openjdk11 repos/trove


bin/activate:
	virtualenv-3 .
	source bin/activate && pip install scikit-learn==0.19.1
	source bin/activate && pip install nltk==3.2.5
	source bin/activate && pip install gensim==3.5.0

# WARNING: This target takes *very* long to complete!
docsim_calc: | docs.db bin/activate
	source bin/activate && python3 calcsim.py
	sqlite3 --batch docs.db \
	    'CREATE UNIQUE INDEX internal_methodsim_idpair ON internal_methodsim (first_id, second_id);'
	@touch docsim_calc


toksim_calc: docsim_calc simtok/build/libs/simtok-all.jar | docs.db 
	java -cp simtok/build/libs/simtok-all.jar CalcSimTok
	@touch toksim_calc


sampled.csv: docsim_calc toksim_calc sample/build/libs/sample-all.jar | docs.db 
	java -Xmx32g -cp sample/build/libs/sample-all.jar Sample 900 sampled.csv


raters:
	python3 -c 'exec("with open(\"raters\", \"w\", encoding=\"utf-8\") as f:\n  while True:\n    rls = input(\"Enter counts for raters separated by spaces: \").split()\n    if all( x.isnumeric() for x in rls ):\n      print(\" \".join(rls), file=f)\n      break")'

html_out: methodextracter/build/libs/methodextracter-all.jar sampled.csv raters
	mkdir -p html_out
	python3 mkhtml.py sampled.csv html_out 3 $(file <raters) > assignments.csv
	@echo "The files for the manual classification are prepared in $@"

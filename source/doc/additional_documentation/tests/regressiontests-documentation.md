Informationen zu den SSB und TPC-H Regressionstests
===================================================

# Aufbau der Klassen

Die Umsetzung der Regressiontests zu den beiden Benchmarks _SSB_ und _TPC-H_
ist auf drei verschiedenen Klassen aufgeteilt:

* **generic_regression_tests**
    * Basisklasse für die beiden folgenden Klassen. Stellt grundlegende
    Funktionalität bereit, die von den beiden Regressiontests genutzt werden.
* **generic_ssb_tests**
    *  Realisiert den Regressionstest für den _Star Schema Benchmark_
* **generic_tpch_tests**
    * Realisiert den Regressionstest für den _TPC-H_ Benchmark


# Testdaten

Die Testdaten zu den Regressionstests sind unter folgendem Pfad zu finden:  `$COGADB/test/testdata/regression test/`

Die Testdaten zum TPC-H Regressionstests befinden sich im Ordner `tpch` und die Testdaten für den Regressionstest zu SSB im Ordner `ssb`.

## TPC-H Testdaten

Die Testdaten zum
_TPC-H_ Benchmark verfügen über den folgenden Aufbau:

```
* results
    * cpu
    * gpu
```

Hier werden die zu erwartenden Ergebnisse für jede Query einzeln aufgeführt. Sie
werden getrennt nach Verwendung der CPU oder der Verwendung der GPU aufgeteilt.

```
* schema
```

Hier wird das zu erwartende Schema der Ergebnis-Tabelle hinterlegt, die bei der Ausführung einer Query erzeugt wird. Das Schema wird als ein SQL
_Create-Statement_ angegeben.

```
* sql
```

Hier sind die SQL-Befehle zu den einzelnen Queries des _TPC-H_ Benchmarks
enthalten. Der Regressionstest berücksichtigt bei Durchführung alle Queries, die
sich in diesem Ordner befinden. Um eine Query temporär von den Tests auszunehmen,
 muss daher nur die Datei aus diesem Ordner entfernt oder in einem Unterordner verschoben werden.


## SSB Testdaten

Die Testdaten zum _Star Schema Benchmark_ verfügen über den folgenden Aufbau:

```
* cpu
* gpu
```

Im Ordner CPU werden sind die Regressionstests bei Verwendung der CPU und im Ornder GPU bei Verwendung der GPU aufgeführt.

Beide Ordner `cpu` und `gpu` besitzen den folgenden Aufbau:

```
* results
```
Hier werden die zu erwartenden Ergebnisse für jede Query einzeln aufgeführt.


```
* schema
```
Hier wird das zu erwartende Schema der Ergebnis-Tabelle hinterlegt, die bei der Ausführung einer Query erzeugt wird. Das Schema wird als ein SQL
_Create-Statement_ angegeben.

```
* sql
```
Hier sind die SQL-Befehle zu den einzelnen Queries des _SSB_ Benchmarks
enthalten. Der Regressionstest berücksichtigt bei Durchführung alle Queries, die
sich in diesem Ordner befinden. Um eine Query temporär von den Tests auszunehmen,
 muss daher nur die Datei aus diesem Ordner entfernt oder in einem Unterordner verschoben werden.


# Weitere Informationen zu den TPC-H Regressionstests

Einige der realisierten Tests zu den einzelnen _TPC-H-Queries_ schlagen derzeit
noch fehl. Die folgenden _TPC-H-Queries_ schlagen derzeit fehl:

* _TPC-H-Query 02_
* _TPC-H-Query 07_
* _TPC-H-Query 15_
* _TPC-H-Query 20_

Die SQL-Dateien zu diesen Queries befinden sich im Ordner `$COGADB/test/testdata/regression test/tpch/`

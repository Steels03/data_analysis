# Rapport

On commence par éditer le nom des colonnes avec la commande suivante :

```
oldColumns = df.schema.names 

df = reduce(lambda df, idx: df.withColumnRenamed(oldColumns[idx], constants.COLUMN_NAMES[idx]), range(len(oldColumns)), df)
```


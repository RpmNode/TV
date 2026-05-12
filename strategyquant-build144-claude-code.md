# StrategyQuant Build 144 — Integración con Claude Code y Análisis Prop Trading

## Los 2 Últimos Videos del Canal StrategyQuant

| # | Título | URL |
|---|--------|-----|
| 1 | StrategyQuant Just Added Claude Code Integration and Prop Firm Analysis (Game Changer) | https://www.youtube.com/watch?v=asYgyb5-VI0 |
| 2 | StrategyQuant 144 integra Claude Code y análisis para prop trading (cambio clave) | https://www.youtube.com/watch?v=NVOAJ0L1Reg |

---

## ¿Qué hay de nuevo en Build 144?

### 1. Integración con Claude Code (Anthropic)
- StrategyQuant X 144 incorpora **Claude Code directamente en la plataforma**, permitiendo que el asistente de IA genere, mejore y depure estrategias de trading escritas en Java/MQL5/EasyLanguage.
- El asistente entiende lenguaje natural: basta describir la estrategia en texto para que genere el código listo para backtest.
- Se pueden dar instrucciones como: *"Añade un filtro de tendencia con EMA 200"* o *"Optimiza el stop loss para que pase el drawdown máximo de la prop firm"*.

### 2. Análisis de Reglas de Prop Trading (Prop Firm Analysis)
- Nuevo módulo que permite **verificar automáticamente** si una estrategia cumple las reglas de empresas de prop trading (FTMO, MyForexFunds, The5ers, etc.).
- Verifica: máximo drawdown diario, drawdown total, profit targets, consistencia, días mínimos de trading.
- Filtra estrategias directamente en el Builder que violarían las reglas configuradas.

---

## Cómo Crear Plugins Personalizados en StrategyQuant con Claude Code

### Arquitectura de Extensión en SQX

StrategyQuant X se extiende mediante **dos tipos de componentes Java**:

| Tipo | Descripción | Ejemplos |
|------|-------------|---------|
| **Snippet** | Función autocontenida que implementa UNA cosa | Indicador, bloque de condición, gestión de dinero |
| **Plugin** | Módulo mayor con UI + lógica de fondo | Monte Carlo, analizador de estrategias, tarea personalizada |

### Requisitos de Entorno

```
- Java JDK 11+
- IntelliJ IDEA (Community o Ultimate)
- StrategyQuant X instalado (para las librerías sqxapi.jar)
- Maven o Gradle para gestión de dependencias
```

### Estructura de un Snippet / Indicador Personalizado

Todo indicador en SQX extiende `IndicatorBlock` e implementa `OnBarUpdate()`:

```java
import com.strategyquant.tradinglib.*;
import com.strategyquant.lib.SQUtils;

@BuildingBlock(
    name     = "Mi Indicador Personalizado",
    display  = "MiIndicador(#Period#)",
    returnType = ReturnTypes.Double
)
public class MiIndicador extends IndicatorBlock {

    // --- Parámetros configurables desde la UI de SQX ---
    @Parameter(defaultValue = "14", minValue = 2, maxValue = 500, step = 1)
    public int Period;

    @Parameter
    public DataSeries Input;

    // --- Buffers de salida ---
    @Output(name = "Main")
    public DataSeries Main;

    // Lógica del indicador (llamado en cada barra)
    @Override
    protected void OnBarUpdate() throws TradingException {
        if (CurrentBar < Period) return;

        double sum = 0;
        for (int i = 0; i < Period; i++) {
            sum += Input.get(i);
        }
        Main.set(sum / Period);
    }
}
```

**Puntos clave:**
- `@BuildingBlock` — registra el bloque en SQX con nombre y tipo de retorno.
- `@Parameter` — expone la variable como parámetro editable en la UI.
- `@Output` — declara el buffer de salida del indicador.
- `OnBarUpdate()` — método obligatorio donde se calcula el valor barra a barra.

---

### Estructura de un Plugin Completo (Tarea Personalizada)

Un plugin de tarea se compone de **dos clases interdependientes**:

```
mi-plugin/
├── src/
│   ├── MiTareaPlugin.java        ← Lógica de la tarea + UI simple
│   └── MiTareaSettingsUI.java    ← UI completa de configuración (AngularJS)
├── pom.xml                       ← Maven config con dependencia a sqxapi.jar
└── resources/
    └── templates/
        └── MiTareaSettings.html  ← Template AngularJS del panel de config
```

**Clase principal de la tarea:**

```java
import com.strategyquant.lib.plugin.*;
import com.strategyquant.tradinglib.*;

@PluginInfo(
    name        = "Mi Tarea Personalizada",
    description = "Analiza estrategias según mis criterios propios",
    category    = "Analysis"
)
public class MiTareaPlugin extends ProjectTask {

    @Override
    public void run(SQProject project) throws Exception {
        // Acceder a las estrategias del proyecto
        List<Strategy> strategies = project.getStrategies();

        for (Strategy s : strategies) {
            double maxDD = s.getStat("MaxDD");
            if (maxDD > 20.0) {
                s.setFlag("FailPropFirm", true);
            }
        }
    }
}
```

---

### Tipos de Extensión Disponibles

```
/user/snippets/          ← Indicadores y bloques personalizados (.java)
/user/plugins/           ← Plugins compilados (.jar)
/user/libs/              ← Librerías JAR externas (ej: TA4J)
/user/strategies/        ← Estrategias generadas
```

Copiar la librería JAR de terceros (ej. `ta4j-core-0.15.jar`) en `/user/libs/` hace que esté disponible automáticamente en todos los snippets.

---

## Flujo de Trabajo con Claude Code para Crear Plugins

### Paso 1 — Describir el plugin en lenguaje natural

Con Build 144, dentro de SQX puedes escribir directamente al asistente Claude:

```
"Crea un indicador de divergencia RSI que detecte divergencias
alcistas y bajistas entre el precio y el RSI de 14 períodos,
con parámetro de umbral configurable."
```

### Paso 2 — Claude genera el snippet Java

Claude Code genera la clase Java con las anotaciones correctas de SQX, lista para compilar y copiar en `/user/snippets/`.

### Paso 3 — Compilar y recargar en SQX

```bash
# Desde IntelliJ o línea de comandos Maven:
mvn clean package

# Copiar el .jar resultante en:
cp target/mi-plugin-1.0.jar "$SQX_HOME/user/plugins/"

# Reinicar StrategyQuant X para cargar el plugin
```

### Paso 4 — Usar Claude Code para analizar prop firm

Una vez cargado el plugin, puedes pedirle a Claude en SQX 144:

```
"Analiza si esta estrategia pasa las reglas de FTMO:
 - Max drawdown diario: 5%
 - Max drawdown total: 10%
 - Profit target: 10%
 - Mínimo 4 días de trading"
```

SQX 144 ejecuta el análisis automáticamente y marca las estrategias que cumplen/fallan.

---

## Recursos Oficiales

| Recurso | URL |
|---------|-----|
| Documentación de programación SQX | https://strategyquant.com/doc/programming-for-sq/ |
| API pública de SQX | https://strategyquant.com/sqxapi/ |
| Guía de extensión (PDF) | https://strategyquant.com/wp-content/uploads/2018/12/Extending_SQX.pdf |
| Indicador ForceIndex (ejemplo completo) | https://strategyquant.com/doc/programming-for-sq/adding-new-indicator-snippet-forceindex/ |
| Indicador Envelopes paso a paso | https://strategyquant.com/doc/programming-for-sq/adding-envelopes-indicator-step-by-step/ |
| Ejemplo plugin tarea completa | https://strategyquant.com/doc/programming-for-sq/example-plugin-a-complete-custom-project-task/ |
| Blog Build 143 (AI Builder) | https://strategyquant.com/blog/strategyquant-build-143-ai-that-builds-your-trading-strategies/ |
| AI Assistant Beta tutorial | https://strategyquant.com/blog/strategyquant-ai-assistant-beta-now-available-full-video-tutorial/ |
| Descargar Dev Build 144 | https://strategyquant.com/download/#tab-dev |

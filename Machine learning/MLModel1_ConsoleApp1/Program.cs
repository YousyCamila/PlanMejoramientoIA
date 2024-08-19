// Este archivo fue autogenerado por ML.NET Model Builder.

using System;
using MLModel1_ConsoleApp1;

class Program
{
    static void Main(string[] args)
    {
        while (true)
        {
            
            Console.WriteLine("Por favor, ingresa un comentario (o escribe 'salir' para terminar):");
            string comentarioUsuario = Console.ReadLine();

           
            if (comentarioUsuario.ToLower() == "salir")
            {
                break;
            }

          
            MLModel1.ModelInput datosEjemplo = new MLModel1.ModelInput()
            {
                Comentario = comentarioUsuario
            };

           
            var prediccion = MLModel1.Predict(datosEjemplo);

            Console.WriteLine("\n=== Resultado de la Predicción ===");
            Console.WriteLine($"Comentario: {datosEjemplo.Comentario}");
            Console.WriteLine($"Etiqueta Predicha: {(prediccion.PredictedLabel == 1 ? "Ofensivo" : "No ofensivo")}\n");

            Console.WriteLine("=============== Fin del proceso ===============\n");
        }
    }
}

package com.example.demo.controller;

import com.example.demo.model.InsuranceRequest;
import com.example.demo.model.InsuranceResponse;
import com.example.demo.service.MLPredictionService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import java.util.Map;
import java.util.Arrays;
import java.util.List;

@RestController
@RequestMapping("/api")
public class PredictController {

    private final MLPredictionService mlPredictionService;
    
    // List of available models
    private static final List<String> AVAILABLE_MODELS = Arrays.asList(
        "random_forest",
        "gradient_boosting",
        "decision_tree",
        "linear_regression",
        "ridge",
        "lasso",
        "elasticnet",
        "svr",
        "knn"
    );
    
    private static final Map<String, String> MODEL_DESCRIPTIONS = Map.ofEntries(
        Map.entry("random_forest", "Random Forest - Ensemble method combining multiple decision trees"),
        Map.entry("gradient_boosting", "Gradient Boosting - Sequential ensemble building on prediction errors"),
        Map.entry("decision_tree", "Decision Tree - Simple tree-based model"),
        Map.entry("linear_regression", "Linear Regression - Simple linear model"),
        Map.entry("ridge", "Ridge Regression - Linear regression with L2 regularization"),
        Map.entry("lasso", "Lasso Regression - Linear regression with L1 regularization"),
        Map.entry("elasticnet", "Elastic Net - Linear regression with combined L1/L2 regularization"),
        Map.entry("svr", "Support Vector Regressor - Non-linear model using support vectors"),
        Map.entry("knn", "K-Nearest Neighbors - Instance-based learning model")
    );

    public PredictController(MLPredictionService mlPredictionService) {
        this.mlPredictionService = mlPredictionService;
    }

    @GetMapping("/models")
    public Map<String, Object> getAvailableModels() {
        return Map.of(
            "available_models", AVAILABLE_MODELS,
            "default_model", "random_forest",
            "descriptions", MODEL_DESCRIPTIONS
        );
    }

    @PostMapping(value = "/predict", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public InsuranceResponse predict(@RequestBody InsuranceRequest req) {
        // Validate required fields
        if (req.getAge() == null || req.getAge() <= 0 || req.getAge() > 120) {
            return new InsuranceResponse("Error: Age must be between 1 and 120");
        }
        if (req.getBmi() == null || req.getBmi() <= 0 || req.getBmi() > 80) {
            return new InsuranceResponse("Error: BMI must be between 1 and 80");
        }
        if (req.getGender() == null || req.getGender().trim().isEmpty()) {
            return new InsuranceResponse("Error: Gender is required");
        }
        if (req.getLocation() == null || req.getLocation().trim().isEmpty()) {
            return new InsuranceResponse("Error: Location is required");
        }
        if (req.getKids() == null || req.getKids() < 0 || req.getKids() > 10) {
            return new InsuranceResponse("Error: Number of kids must be between 0 and 10");
        }

        // Validate new fields with defaults
        String income = (req.getIncome() != null && !req.getIncome().trim().isEmpty()) ? req.getIncome() : "medium";
        String employment = (req.getEmployment() != null && !req.getEmployment().trim().isEmpty()) ? req.getEmployment() : "employed";
        Integer healthScore = (req.getHealthScore() != null) ? req.getHealthScore() : 7;
        Integer exerciseFrequency = (req.getExerciseFrequency() != null) ? req.getExerciseFrequency() : 3;
        String education = (req.getEducation() != null && !req.getEducation().trim().isEmpty()) ? req.getEducation() : "bachelor";
        String maritalStatus = (req.getMaritalStatus() != null && !req.getMaritalStatus().trim().isEmpty()) ? req.getMaritalStatus() : "single";
        Integer yearsInsured = (req.getYearsInsured() != null) ? req.getYearsInsured() : 0;

        // Validate ranges
        if (healthScore < 1 || healthScore > 10) {
            return new InsuranceResponse("Error: Health score must be between 1 and 10");
        }
        if (exerciseFrequency < 0 || exerciseFrequency > 7) {
            return new InsuranceResponse("Error: Exercise frequency must be between 0 and 7 days per week");
        }
        if (yearsInsured < 0 || yearsInsured > 50) {
            return new InsuranceResponse("Error: Years insured must be between 0 and 50");
        }

        // Validate enum-like values
        String[] validIncomes = {"low", "medium", "high", "very_high"};
        if (!java.util.Arrays.asList(validIncomes).contains(income.toLowerCase())) {
            return new InsuranceResponse("Error: Income must be one of: low, medium, high, very_high");
        }

        String[] validEmployments = {"employed", "self_employed", "unemployed", "retired"};
        if (!java.util.Arrays.asList(validEmployments).contains(employment.toLowerCase())) {
            return new InsuranceResponse("Error: Employment must be one of: employed, self_employed, unemployed, retired");
        }

        String[] validEducations = {"high_school", "bachelor", "master", "phd"};
        if (!java.util.Arrays.asList(validEducations).contains(education.toLowerCase())) {
            return new InsuranceResponse("Error: Education must be one of: high_school, bachelor, master, phd");
        }

        String[] validMarital = {"single", "married", "divorced", "widowed"};
        if (!java.util.Arrays.asList(validMarital).contains(maritalStatus.toLowerCase())) {
            return new InsuranceResponse("Error: Marital status must be one of: single, married, divorced, widowed");
        }

        String[] validLocations = {"northeast", "southeast", "southwest", "northwest"};
        if (!java.util.Arrays.asList(validLocations).contains(req.getLocation().toLowerCase())) {
            return new InsuranceResponse("Error: Location must be one of: northeast, southeast, southwest, northwest");
        }

        // Get ML model prediction with selected model
        String selectedModel = req.getModel();
        if (selectedModel == null || selectedModel.trim().isEmpty()) {
            selectedModel = "random_forest"; // Default model
        }
        
        Map<String, Object> predictionResult = mlPredictionService.predictInsuranceCost(
                req.getAge(),
                req.getGender(),
                req.getBmi(),
                req.getKids(),
                req.getSmoker() != null ? req.getSmoker() : false,
                req.getLocation(),
                income,
                employment,
                healthScore,
                exerciseFrequency,
                education,
                maritalStatus,
                yearsInsured,
                selectedModel
        );
        
        String usedModel = (String) predictionResult.get("model");
        double predictedCost = ((Number) predictionResult.get("prediction")).doubleValue();

        // Generate a friendly response with suggestions
        StringBuilder response = new StringBuilder();
        response.append("=== Insurance Cost Estimate ===\n");
        response.append(String.format("Model Used: %s\n", usedModel.replace("_", " ").toUpperCase()));
        response.append(String.format("Estimated Annual Cost: $%.2f\n\n", predictedCost));
        
        response.append("=== Your Profile ===\n");
        response.append(String.format("Age: %d | BMI: %.1f | Gender: %s\n", req.getAge(), req.getBmi(), req.getGender()));
        response.append(String.format("Smoker: %s | Children: %d | Location: %s\n", 
            (req.getSmoker() != null && req.getSmoker()) ? "Yes" : "No", req.getKids(), req.getLocation()));
        response.append(String.format("Income: %s | Employment: %s\n", income, employment));
        response.append(String.format("Health Score: %d/10 | Exercise: %d days/week | Education: %s\n", 
            healthScore, exerciseFrequency, education));
        response.append(String.format("Marital Status: %s | Years Insured: %d\n\n", maritalStatus, yearsInsured));
        
        response.append("=== Personalized Suggestions to Reduce Cost ===\n");
        
        // Generate suggestions based on input
        if (req.getSmoker() != null && req.getSmoker()) {
            response.append("1. Consider quitting smoking - this could reduce your premium by up to 50%\n");
        } else {
            response.append("1. Maintain your non-smoker status - this keeps your premiums lower\n");
        }

        if (req.getBmi() != null && req.getBmi() > 30) {
            response.append("2. Work on reducing your BMI through diet and exercise - lower BMI means lower costs\n");
        } else if (req.getBmi() != null && req.getBmi() > 25) {
            response.append("2. Maintain a healthy BMI to keep insurance costs stable\n");
        } else {
            response.append("2. Keep maintaining your healthy BMI - this is excellent for your health and insurance costs\n");
        }

        if (healthScore < 6) {
            response.append("3. Improve your health score by regular check-ups and preventive care\n");
        } else if (healthScore >= 8) {
            response.append("3. Excellent health score! Keep up with your wellness routine\n");
        }

        if (exerciseFrequency < 3) {
            response.append("4. Increase exercise frequency - aim for at least 3-5 days per week for better health\n");
        } else {
            response.append("4. Excellent exercise routine! Keep maintaining your fitness level\n");
        }

        if (education.equalsIgnoreCase("high_school")) {
            response.append("5. Consider pursuing higher education for better career and health insurance prospects\n");
        }

        response.append("6. Schedule regular health check-ups to prevent chronic conditions\n");

        if (req.getExistingCondition() != null && !req.getExistingCondition().trim().isEmpty()) {
            response.append("\nNote: Your existing condition(s) may impact the final premium. Please consult with an insurance agent for detailed information.\n");
        }

        return new InsuranceResponse(response.toString());
    }
}

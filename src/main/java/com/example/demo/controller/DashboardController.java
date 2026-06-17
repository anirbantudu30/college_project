package com.example.demo.controller;

import com.example.demo.dto.HealthProfileDto;
import com.example.demo.dto.RecommendedDoctorDto;
import com.example.demo.dto.RecommendedHospitalDto;
import com.example.demo.entity.User;
import com.example.demo.entity.pastPredictions;
import com.example.demo.repository.UserRepository;
import com.example.demo.service.HealthProfileService;
import com.example.demo.service.RecommendationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;
import com.example.demo.service.UserService;
import com.example.demo.repository.PastPredictionsRepository;

import java.util.List;

@Controller
@RequestMapping("/dashboard")
public class DashboardController {

    @Autowired
    private PastPredictionsRepository pastPredictionsRepository;
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private HealthProfileService healthProfileService;
    
    @Autowired
    private RecommendationService recommendationService;

    @Autowired
    private UserService userService;
    
    @GetMapping
    public String dashboard(Model model,Authentication authentication) {
        User user = getCurrentUser();
        if (user == null) {
            return "redirect:/login";
        }
        
        // Get health profile
        HealthProfileDto healthProfile = healthProfileService.getHealthProfileDto(user);
        model.addAttribute("healthProfile", healthProfile);
        model.addAttribute("predictCount",userService.findByUsername(authentication.getName()).get().getPredictCount());
        System.out.println("Predict Count for user " + authentication.getName() + ": " + userService.findByUsername(authentication.getName()).get().getPredictCount());
        
        // Get recommendations
        List<RecommendedHospitalDto> hospitals = recommendationService.getRecommendedHospitals(user);
        List<RecommendedDoctorDto> doctors = recommendationService.getRecommendedDoctors(user); 
        
        model.addAttribute("username", user.getUsername());
        model.addAttribute("recommendedHospitals", hospitals);
        model.addAttribute("recommendedDoctors", doctors);
        model.addAttribute("hasConditions", !hospitals.isEmpty() || !doctors.isEmpty());
        
        return "dashboard";
    }
    
    @GetMapping("/health")
    public String healthProfile(Model model) {
        User user = getCurrentUser();
        if (user == null) {
            return "redirect:/login";
        }
        
        HealthProfileDto healthProfile = healthProfileService.getHealthProfileDto(user);
        model.addAttribute("healthProfile", healthProfile);
        model.addAttribute("username", user.getUsername());
        
        return "health-profile";
    }
    
    @PostMapping("/health/update")
    public String updateHealthProfile(@ModelAttribute HealthProfileDto healthProfileDto,
                                      RedirectAttributes redirectAttributes) {
        User user = getCurrentUser();
        if (user == null) {
            return "redirect:/login";
        }
        
        try {
            healthProfileService.updateHealthProfile(user, healthProfileDto);
            redirectAttributes.addFlashAttribute("success", "Health profile updated successfully!");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to update health profile: " + e.getMessage());
        }
        
        return "redirect:/dashboard";
    }
    
    @GetMapping("/recommendations")
    public String recommendations(Model model) {
        User user = getCurrentUser();
        if (user == null) {
            return "redirect:/login";
        }
        
        List<RecommendedHospitalDto> hospitals = recommendationService.getRecommendedHospitals(user);
        List<RecommendedDoctorDto> doctors = recommendationService.getRecommendedDoctors(user);
        
        model.addAttribute("username", user.getUsername());
        model.addAttribute("recommendedHospitals", hospitals);
        model.addAttribute("recommendedDoctors", doctors);
        model.addAttribute("hasConditions", !hospitals.isEmpty() || !doctors.isEmpty());
        
        return "recommendations";
    }

    @GetMapping("/history")
    public String predictionHistory(Model model,Authentication authentication) {
        User user=userService.findByUsername(authentication.getName()).get();
        List<pastPredictions> pastPredictions = pastPredictionsRepository.findByUserId(user.getId());
        model.addAttribute("pastPredictions", pastPredictions);
        return "history";

    }

    
    private User getCurrentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.isAuthenticated() && !auth.getName().equals("anonymousUser")) {
            return userRepository.findByUsername(auth.getName()).orElse(null);
        }
        return null;
    }
}

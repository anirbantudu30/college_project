package com.example.demo.entity;

import jakarta.persistence.*;


@Entity
@Table(name = "past_predictions")
public class pastPredictions {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id", nullable = false)
    private Long id;
    
    @Column
    private Double estimatedCost;

    @Column(length=5000)
    private String advice;

    @Column
    private Long userId;

    @Column(length=6000)
    private String profile;

    @Column
    private String modelUsed;

    public pastPredictions() {
    }
    public pastPredictions(Double estimatedCost, String advice, Long userId, String profile, String modelUsed) {
        this.estimatedCost = estimatedCost;
        this.advice = advice;
        this.userId = userId;
        this.profile = profile;
        this.modelUsed = modelUsed;
    }

    public Long getId() {
        return id;
    }
    public void setId(Long id) {
        this.id = id;
    }

    public Double getEstimatedCost() {
        return estimatedCost;
    }
    public void setEstimatedCost(Double estimatedCost) {
        this.estimatedCost = estimatedCost;
    }
    public String getAdvice() {
        return advice;
    }
    public void setAdvice(String advice) {
        this.advice = advice;
    }
    public Long getUserId() {
        return userId;
    }
    public void setUserId(Long userId) {
        this.userId = userId;
    }
    public String getProfile() {
        return profile;
    }
    public void setProfile(String profile) {
        this.profile = profile;
    }
    public String getModelUsed() {
        return modelUsed;
    }
    public void setModelUsed(String modelUsed) {
        this.modelUsed = modelUsed;
    }
}

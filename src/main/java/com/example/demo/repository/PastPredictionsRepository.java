package com.example.demo.repository;
import com.example.demo.entity.pastPredictions;
import org.springframework.stereotype.Repository;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;


@Repository
public interface PastPredictionsRepository extends JpaRepository<pastPredictions, Long> {
    List<pastPredictions> findByUserId(Long userId);
    
}

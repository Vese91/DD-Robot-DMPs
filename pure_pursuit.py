''' 
Code that implements the pure pursuit algorithm for a differential-drive robot to follow a path. 
The code calculates the goal point for the robot to aim to reach, the index of the last found 
point in the path, and the turning velocity that allows the robot to reach the goal point. 
The robot linear velocity is assumed to be constant or can be calculated using a PID controller.

For further details, please refer to: 
https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit
'''

import numpy as np
import matplotlib.pyplot as plt
import math as math

def sgn(number):
    # Returns -1 if number is negative, 1 otherwise
    if number >= 0:
        return 1
    else:
        return -1
    
def pt_to_pt_distance(pt1,pt2):
    # Returns the Euclidean distance between two points
    distance = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
    return distance

def pure_pursuit_step(path, currentPos, currentHeading, lookAheadDis, LFindex):
  '''
  Pure pursuit algorithm for a robot to follow a path.
  
  Inputs:
  path - list of points that the robot should follow.
  currentPos - list of the current x and y position of the robot.
  currentHeading - the current heading of the robot in degrees.  
  lookAheadDis - the distance the robot should look ahead to find the goal point.
  LFindex - the index of the last found point in the path.
  
  Outputs:
  goalPt - the point the robot should aim to reach.
  lastFoundIndex - the index of the last found point in the path.
  turnVel - the turning velocity of the robot.
  '''

  if LFindex == len(path)-1:
    goalPt = path[-1]
    lastFoundIndex = LFindex
    # Obtained goal point, now compute turn vel initialize proportional controller constant
    Kp = 3

    # Calculate absTargetAngle with the atan2 function 
    # absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0]) *180/np.pi
    absTargetAngle = math.atan2(goalPt[1]-currentPos[1], goalPt[0]-currentPos[0])
    if absTargetAngle < 0: 
      absTargetAngle += 2*np.pi

    # Compute turn error by finding the minimum angle
    turnError = absTargetAngle - currentHeading
    if turnError > np.pi or turnError < -np.pi:
      turnError = -1 * sgn(turnError) * (2*np.pi - abs(turnError))

    # Apply proportional controller
    turnVel = Kp*turnError
    return goalPt, lastFoundIndex, turnVel
  else:
    # extract currentX and currentY
    currentX = currentPos[0]
    currentY = currentPos[1]

    # use for loop to search intersections
    lastFoundIndex = LFindex
    intersectFound = False
    startingIndex = lastFoundIndex

    for i in range(startingIndex, len(path)-1):
       # beginning of line-circle intersection code
      x1 = path[i][0] - currentX
      y1 = path[i][1] - currentY
      x2 = path[i+1][0] - currentX
      y2 = path[i+1][1] - currentY
      dx = x2 - x1
      dy = y2 - y1
      dr = math.sqrt (dx**2 + dy**2)
      D = x1*y2 - x2*y1
      discriminant = (lookAheadDis**2) * (dr**2) - D**2

      if discriminant >= 0:
        # Solutions exist
        sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
        sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
        sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
        sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

        sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
        sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]

        # End of line-circle intersection code
        min_x = min(path[i][0], path[i+1][0])
        min_y = min(path[i][1], path[i+1][1])
        max_x = max(path[i][0], path[i+1][0])
        max_y = max(path[i][1], path[i+1][1])

        # If one or both solutions are in range
        if ((min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y)) and ((min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y)):
           foundIntersection = True
           # If both solutions are in range, check which one is better
           if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
            goalPt = sol_pt1
           else:
            goalPt = sol_pt2
        
        # If not both solutions are in range, take the one that's in range
        else:
           # If solution pt1 is in range, take the one that's in range
           if (min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y):
            goalPt = sol_pt1
           else:
            goalPt = sol_pt2

        # Only exit loop if the solution pt found is closer to the next pt in path than the current pos
        if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
          # update lastFoundIndex and exit
          lastFoundIndex = i
          break
        else:
          # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
          lastFoundIndex = i+1

      # If no solutions are in range
      else:
        foundIntersection = False
        # No new intersection found, potentially deviated from the path
        # Follow path[lastFoundIndex]
        goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]
    
    # Obtained goal point, now compute turn vel initialize proportional controller constant
    Kp = 3

    # Calculate absTargetAngle with the atan2 function 
    # absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0]) *180/np.pi
    absTargetAngle = math.atan2(goalPt[1]-currentPos[1], goalPt[0]-currentPos[0])
    if absTargetAngle < 0: 
      absTargetAngle += 2*np.pi

    # Compute turn error by finding the minimum angle
    turnError = absTargetAngle - currentHeading
    if turnError > np.pi or turnError < -np.pi:
      turnError = -1 * sgn(turnError) * (2*np.pi - abs(turnError))

    # Apply proportional controller
    turnVel = Kp*turnError
  
    return goalPt, lastFoundIndex, turnVel

# def pure_pursuit_step(path, currentPos, currentHeading, lookAheadDis, LFindex):
#   '''
#   Pure pursuit algorithm for a robot to follow a path.
  
#   Inputs:
#   path - list of points that the robot should follow.
#   currentPos - list of the current x and y position of the robot.
#   currentHeading - the current heading of the robot in degrees.  
#   lookAheadDis - the distance the robot should look ahead to find the goal point.
#   LFindex - the index of the last found point in the path.
  
#   Outputs:
#   goalPt - the point the robot should aim to reach.
#   lastFoundIndex - the index of the last found point in the path.
#   turnVel - the turning velocity of the robot.
#   '''

#   # extract currentX and currentY
#   currentX = currentPos[0]
#   currentY = currentPos[1]

#   # use for loop to search intersections
#   lastFoundIndex = LFindex
#   intersectFound = False
#   startingIndex = lastFoundIndex
  
#   for i in range(startingIndex, len(path)-1):

#     # beginning of line-circle intersection code
#     x1 = path[i][0] - currentX
#     y1 = path[i][1] - currentY
#     x2 = path[i+1][0] - currentX
#     y2 = path[i+1][1] - currentY
#     dx = x2 - x1
#     dy = y2 - y1
#     dr = math.sqrt (dx**2 + dy**2)
#     D = x1*y2 - x2*y1
#     discriminant = (lookAheadDis**2) * (dr**2) - D**2

#     if discriminant >= 0:
#       # Solutions exist
#       sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
#       sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
#       sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
#       sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

#       sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
#       sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]

#       # end of line-circle intersection code
#       min_x = min(path[i][0], path[i+1][0])
#       min_y = min(path[i][1], path[i+1][1])
#       max_x = max(path[i][0], path[i+1][0])
#       max_y = max(path[i][1], path[i+1][1])

#       # if one or both of the solutions are in range
#       if ((min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y)) or ((min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y)):

#         foundIntersection = True

#         # if both solutions are in range, check which one is better
#         if ((min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y)) and ((min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y)):
#           # make the decision by compare the distance between the intersections and the next point in path
#           if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
#             goalPt = sol_pt1
#           else:
#             goalPt = sol_pt2
        
#         # if not both solutions are in range, take the one that's in range
#         else:
#           # if solution pt1 is in range, set that as goal point
#           if (min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y):
#             goalPt = sol_pt1
#           else:
#             goalPt = sol_pt2
          
#         # only exit loop if the solution pt found is closer to the next pt in path than the current pos
#         if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
#           # update lastFoundIndex and exit
#           lastFoundIndex = i
#           break
#         else:
#           # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
#           lastFoundIndex = i+1
        
#       # if no solutions are in range
#       else:
#         foundIntersection = False
#         # no new intersection found, potentially deviated from the path
#         # follow path[lastFoundIndex]
#         goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

#   # obtained goal point, now compute turn vel
#   # initialize proportional controller constant
#   Kp = 3

#   # calculate absTargetAngle with the atan2 function
#   # absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0]) *180/np.pi
#   absTargetAngle = math.atan2(goalPt[1]-currentPos[1], goalPt[0]-currentPos[0])
#   if absTargetAngle < 0: 
#     absTargetAngle += 2*np.pi

#   # compute turn error by finding the minimum angle
#   turnError = absTargetAngle - currentHeading
#   if turnError > np.pi or turnError < -np.pi:
#     turnError = -1 * sgn(turnError) * (2*np.pi - abs(turnError))
  
#   # apply proportional controller
#   turnVel = Kp*turnError
  
#   return goalPt, lastFoundIndex, turnVel